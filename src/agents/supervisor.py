"""
Supervisor Agent pour AIPROD
Agent d'approbation finale - Certifie la qualité et le coût pour la livraison
"""
import asyncio
from typing import Any, Dict
from src.utils.monitoring import logger


class Supervisor:
    """
    Agent Supervisor pour AIPROD.
    Vérifie la qualité et le coût avant approbation finale.
    Génère le manifeste de livraison.
    """

    def __init__(self):
        self.name = "AIPROD Supervisor"
        self.llm_model = "gemini-1.5-pro"
        logger.info(f"Supervisor initialized: {self.name}")

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute la logique de supervision finale.
        
        Args:
            inputs: Dict contenant:
                - consistency_report: Rapport de QA sémantique
                - cost_certification: Certification des coûts
                - technical_validation_report: Rapport QA technique
                - quality_score: Score de qualité
                - client_budget: Budget du client (optionnel)
        
        Returns:
            Dict avec:
                - final_approval: True/False
                - delivery_manifest: Manifeste de livraison
                - quality_certification: Certification de qualité
                - client_report: Rapport pour le client
        """
        try:
            logger.info("Supervisor: Starting final approval process")

            # Extraction des inputs
            consistency_report = inputs.get("consistency_report", {})
            cost_certification = inputs.get("cost_certification", {})
            technical_report = inputs.get("technical_validation_report", {})
            quality_score = inputs.get("quality_score", 0.5)
            client_budget = inputs.get("client_budget", 1000.0)
            estimated_cost = cost_certification.get("total_cost", 0.0)

            # Extraction des scores
            technical_score = technical_report.get("technical_score", 0.8)

            # Matrice de décision
            approval = self._decision_matrix(
                quality_score=quality_score,
                technical_score=technical_score,
                estimated_cost=estimated_cost,
                client_budget=client_budget
            )

            logger.info(f"Supervisor: Decision - {approval['decision']}")

            # Génération du manifeste de livraison
            delivery_manifest = {
                "version": "1.0",
                "approved": approval["decision"] == "APPROVED",
                "quality_score": quality_score,
                "technical_score": technical_score,
                "estimated_cost": estimated_cost,
                "client_budget": client_budget,
                "consistency_report": consistency_report,
                "cost_certification": cost_certification,
                "decision_reason": approval["reason"],
                "timestamp": self._get_timestamp()
            }

            # Rapport pour le client
            client_report = {
                "status": "APPROVED" if approval["decision"] == "APPROVED" else "PENDING_REVIEW",
                "quality_metrics": {
                    "quality_score": quality_score,
                    "technical_score": technical_score,
                    "consistency_level": consistency_report.get("consistency_level", 0.0)
                },
                "cost_summary": {
                    "estimated_total": estimated_cost,
                    "budget": client_budget,
                    "within_budget": estimated_cost <= client_budget,
                    "breakdown": cost_certification.get("breakdown", {})
                },
                "next_steps": approval["next_steps"],
                "message": approval["message"]
            }

            result = {
                "final_approval": approval["decision"] == "APPROVED",
                "delivery_manifest": delivery_manifest,
                "quality_certification": {
                    "quality_score": quality_score,
                    "technical_score": technical_score,
                    "certified": approval["decision"] == "APPROVED"
                },
                "client_report": client_report,
                "decision": approval["decision"]
            }

            logger.info(f"Supervisor: Final approval result - {result['final_approval']}")
            return result

        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            return {
                "final_approval": False,
                "error": str(e),
                "decision": "ERROR",
                "client_report": {"status": "ERROR", "message": f"Supervision failed: {e}"}
            }

    def _decision_matrix(
        self,
        quality_score: float,
        technical_score: float,
        estimated_cost: float,
        client_budget: float
    ) -> Dict[str, Any]:
        """
        Matrice de décision basée sur les seuils définis dans le JSON.
        
        Règles:
        - APPROVED si: quality_score >= 0.7 ET estimated_cost <= client_budget
        - REJECTED si: quality_score < 0.4 OU technical_score < 0.8
        - ESCALATE si: quality_score entre 0.4 et 0.7
        """
        # Vérification des conditions de rejet
        if quality_score < 0.4 or technical_score < 0.8:
            return {
                "decision": "REJECTED",
                "reason": f"Quality (${quality_score:.2f}) or Technical ({technical_score:.2f}) below threshold",
                "next_steps": ["Review quality issues", "Re-render or re-evaluate"],
                "message": "The pipeline quality does not meet minimum requirements. Please review and retry."
            }

        # Escalade
        if 0.4 <= quality_score < 0.7:
            return {
                "decision": "ESCALATE",
                "reason": f"Quality score {quality_score:.2f} in escalation range (0.4-0.7)",
                "next_steps": ["Manual review required", "Client approval needed"],
                "message": "This project requires manual review before approval. Please confirm with the team."
            }

        # Approbation
        if quality_score >= 0.7 and estimated_cost <= client_budget:
            return {
                "decision": "APPROVED",
                "reason": f"Quality ({quality_score:.2f}) and Cost (${estimated_cost:.2f} <= ${client_budget:.2f}) acceptable",
                "next_steps": ["Prepare delivery", "Upload to storage", "Send to client"],
                "message": "Project approved for delivery!"
            }

        # Budget exceeded mais qualité OK
        if quality_score >= 0.7 and estimated_cost > client_budget:
            budget_overrun = ((estimated_cost - client_budget) / client_budget * 100)
            return {
                "decision": "REVIEW",
                "reason": f"Budget exceeded by {budget_overrun:.1f}% (${estimated_cost:.2f} > ${client_budget:.2f})",
                "next_steps": ["Request budget increase", "Offer cost optimization", "Client confirmation"],
                "message": f"Project quality approved but exceeds budget by {budget_overrun:.1f}%. Client approval required."
            }

        return {
            "decision": "UNKNOWN",
            "reason": "Unknown decision path",
            "next_steps": ["Debug decision matrix"],
            "message": "An unexpected state was reached. Please review the decision matrix."
        }

    @staticmethod
    def _get_timestamp() -> str:
        """Récupère le timestamp courant."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
