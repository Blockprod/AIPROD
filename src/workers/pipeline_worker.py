"""
Pipeline Worker - Consomme les jobs depuis Pub/Sub et les exécute.

P1.2.3: Background worker qui:
1. Consomme les messages JobMessage depuis aiprod-pipeline-jobs subscription
2. Exécute le pipeline via state_machine.run()
3. Publie les résultats vers aiprod-pipeline-results
4. Met à jour le job status en PostgreSQL
5. Gère les erreurs et les envoie vers DLQ
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
from concurrent import futures
from functools import wraps

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import pubsub_v1
from google.api_core.exceptions import GoogleAPICallError

from src.orchestrator.state_machine import StateMachine
from src.db.job_repository import JobRepository
from src.db.models import get_session_factory, JobState
from src.pubsub.client import get_pubsub_client, JobMessage, ResultMessage
from src.api.functions.input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Worker qui traite les jobs depuis Pub/Sub."""

    def __init__(self, project_id: Optional[str] = None, num_threads: int = 5):
        """
        Initialise le worker.

        Args:
            project_id: GCP project ID (defaut: env var GOOGLE_CLOUD_PROJECT)
            num_threads: Nombre de threads pour traiter les messages
        """
        self.project_id = project_id or os.getenv(
            "GOOGLE_CLOUD_PROJECT", "aiprod-484120"
        )
        self.num_threads = num_threads

        # Clients
        self.pubsub_client = get_pubsub_client()
        self.subscriber = pubsub_v1.SubscriberClient()

        # Database
        self.db_url = os.getenv(
            "DATABASE_URL", "postgresql://aiprod:password@localhost:5432/aiprod_v33"
        )
        session_factory, _ = get_session_factory(self.db_url)
        self.session_factory = session_factory

        # State machine et sanitizer
        self.state_machine = StateMachine()
        self.input_sanitizer = InputSanitizer()

        # Subscription path
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, "aiprod-pipeline-jobs-sub"
        )

        logger.info(
            f"🚀 PipelineWorker initialized (project={self.project_id}, threads={num_threads})"
        )

    def process_message(self, message: pubsub_v1.subscriber.message.Message) -> bool:  # type: ignore[misc]
        """
        Traite un message de job depuis Pub/Sub.

        Args:
            message: Message Pub/Sub contenant un JobMessage

        Returns:
            bool: True si succès, False si erreur
        """
        start_time = time.time()
        job_id = None

        try:
            # 1. Décoder le message
            data = json.loads(message.data.decode("utf-8"))
            job_msg = JobMessage.from_dict(data)
            job_id = job_msg.job_id

            if not job_id:
                logger.error("❌ Job ID is None, cannot process message")
                message.nack()
                return False

            logger.info(
                f"📨 Processing job {job_id} for user {job_msg.user_id} "
                f"with preset {job_msg.preset}"
            )

            # 2. Mettre à jour le job status → PROCESSING
            db_session = self.session_factory()
            try:
                job_repo = JobRepository(db_session)
                job_repo.update_job_state(
                    job_id, JobState.PROCESSING, reason="Worker starting processing"
                )
                logger.info(f"✅ Job {job_id} state → PROCESSING")
            finally:
                db_session.close()

            # 3. Exécuter le pipeline
            logger.info(f"⚙️  Running pipeline for job {job_id}...")

            # Préparer les données d'entrée
            input_data = {
                "content": job_msg.content,
                "preset": job_msg.preset,
                "_user_id": job_msg.user_id,
                "_job_id": job_id,
            }

            # Ajouter les métadonnées
            if job_msg.metadata:
                input_data.update(job_msg.metadata)

            # Sanitize inputs
            sanitized = self.input_sanitizer.sanitize(input_data)

            # Exécuter le state machine (async pipeline)
            result = asyncio.run(self.state_machine.run(sanitized))

            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Pipeline completed for {job_id} in {execution_time_ms}ms")

            # 4. Mettre à jour le job avec le résultat
            db_session = self.session_factory()
            try:
                job_repo = JobRepository(db_session)

                # Récupérer le job
                job = job_repo.get_job(job_id)
                if not job:
                    logger.error(f"❌ Job {job_id} not found after processing")
                    return False

                # Ajouter le résultat
                job_repo.set_job_result(
                    job_id=job_id,
                    status="success",
                    output=result,
                    error_message=None,
                    processing_time_ms=execution_time_ms,
                )

                # Mettre à jour le statut
                job_repo.update_job_state(
                    job_id,
                    JobState.COMPLETED,
                    reason="Pipeline execution completed successfully",
                )

                logger.info(f"✅ Job {job_id} status → COMPLETED")
            finally:
                db_session.close()

            # 5. Publier le résultat vers Pub/Sub
            # (Message créé directement dans publish_result)

            result_msg_id = self.pubsub_client.publish_result(
                job_id=job_id,
                status="success",
                output=result,
                error_message=None,
                processing_time_ms=execution_time_ms,
            )

            logger.info(f"📤 Result published for {job_id} (msg_id={result_msg_id})")

            # 6. Acknowledger le message
            message.ack()
            logger.info(f"✅ Message acknowledged for job {job_id}")

            return True

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"❌ Error processing job {job_id}: {str(e)}", exc_info=True)

            if job_id:
                try:
                    # Mettre à jour le job comme FAILED
                    db_session = self.session_factory()
                    try:
                        job_repo = JobRepository(db_session)
                        job_repo.update_job_state(
                            job_id,
                            JobState.FAILED,
                            reason=f"Pipeline execution error: {str(e)}",
                        )
                        job_repo.set_job_result(
                            job_id=job_id,
                            status="error",
                            output=None,
                            error_message=str(e),
                            processing_time_ms=execution_time_ms,
                        )
                    finally:
                        db_session.close()
                except Exception as db_error:
                    logger.error(f"❌ Failed to update job status: {str(db_error)}")

                try:
                    # Publier vers DLQ
                    dlq_msg_id = self.pubsub_client.publish_dlq_message(
                        job_id=job_id,
                        reason="Pipeline execution error",
                        error=str(e),
                        metadata={"preset": "unknown"},
                    )
                    logger.warning(
                        f"⚠️  Job {job_id} published to DLQ (msg_id={dlq_msg_id})"
                    )
                except Exception as dlq_error:
                    logger.error(f"❌ Failed to publish to DLQ: {str(dlq_error)}")

            # Nacker le message (renvoyer à la queue pour retry)
            message.nack()
            logger.warning(f"⚠️  Message nacked for job {job_id} (will be retried)")

            return False

    def start(self):
        """Démarre le worker - consomme les messages indéfiniment."""
        logger.info(f"🚀 Starting worker listening on {self.subscription_path}...")

        # Créer un streaming pull future
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.process_message,
            flow_control=pubsub_v1.types.FlowControl(
                max_messages=self.num_threads, max_bytes=10 * 1024 * 1024  # 10MB
            ),
        )

        logger.info("✅ Worker ready. Processing messages...")

        try:
            # Attendre indéfiniment
            streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down worker...")
            streaming_pull_future.cancel()
            streaming_pull_future.result()
            logger.info("✅ Worker shutdown complete")
        except Exception as e:
            logger.error(f"❌ Worker error: {str(e)}")
            streaming_pull_future.cancel()
            streaming_pull_future.result()


def main():
    """Point d'entrée principal du worker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AIPROD V33 Pipeline Worker - Background job processor"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP Project ID (default: env var GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--threads", type=int, default=5, help="Number of worker threads (default: 5)"
    )

    args = parser.parse_args()

    # Initialiser le worker
    worker = PipelineWorker(project_id=args.project, num_threads=args.threads)

    # Démarrer
    worker.start()


if __name__ == "__main__":
    main()
