"""
Load Testing Script - Performance & Stress Testing
==================================================

Comprehensive load testing for AIPROD API:
- Concurrent request simulation
- End-to-end pipeline testing
- Performance metrics collection
- Stress testing under load
- Error rate monitoring

PHASE 4 implementation (Weeks 11-13).

Usage:
    python scripts/load_test.py --requests 100 --concurrency 10
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import argparse
import json


class LoadTester:
    """
    Load testing harness for AIPROD API.
    
    Simulates concurrent user requests and measures:
    - Response time (p50, p95, p99)
    - Throughput (requests/sec)
    - Error rate
    - Pipeline completion rate
    - Cost per request
    """
    
    def __init__(
        self,
        base_url: str,
        num_requests: int,
        concurrency: int
    ):
        """Initialize load tester."""
        self.base_url = base_url
        self.num_requests = num_requests
        self.concurrency = concurrency
        
        # Results
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
        print(f"LoadTester initialized: {num_requests} requests, concurrency={concurrency}")
    
    async def run(self):
        """
        Run load test.
        
        Returns:
            Test results summary
        """
        print(f"\n{'='*60}")
        print(f"Starting load test: {self.num_requests} requests")
        print(f"Concurrency: {self.concurrency}")
        print(f"Target: {self.base_url}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Create tasks
        tasks = [
            self._execute_request(request_id, semaphore)
            for request_id in range(self.num_requests)
        ]
        
        # Execute concurrently
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate report
        report = self._generate_report(total_duration)
        
        return report
    
    async def _execute_request(self, request_id: int, semaphore: asyncio.Semaphore):
        """
        Execute single API request.
        
        Args:
            request_id: Request identifier
            semaphore: Concurrency control semaphore
        """
        async with semaphore:
            request_start = time.time()
            
            try:
                # Create request payload
                payload = self._generate_payload(request_id)
                
                # Make API request
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        
                        response_data = await response.json()
                        request_end = time.time()
                        
                        # Record result
                        result = {
                            "request_id": request_id,
                            "status_code": response.status,
                            "duration": request_end - request_start,
                            "timestamp": datetime.now().isoformat(),
                            "success": response.status == 200,
                            "cost": response_data.get("cost_estimation", {}).get("total_estimated", 0),
                            "quality": response_data.get("semantic_validation_report", {}).get("average_score", 0)
                        }
                        
                        self.results.append(result)
                        
                        # Progress indicator
                        if (request_id + 1) % 10 == 0:
                            print(f"Completed: {request_id + 1}/{self.num_requests}")
            
            except asyncio.TimeoutError:
                request_end = time.time()
                
                error = {
                    "request_id": request_id,
                    "error": "timeout",
                    "duration": request_end - request_start,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.errors.append(error)
                print(f"Request {request_id} timed out")
            
            except Exception as e:
                request_end = time.time()
                
                error = {
                    "request_id": request_id,
                    "error": str(e),
                    "duration": request_end - request_start,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.errors.append(error)
                print(f"Request {request_id} failed: {e}")
    
    def _generate_payload(self, request_id: int) -> Dict[str, Any]:
        """
        Generate test request payload.
        
        Args:
            request_id: Request identifier
            
        Returns:
            API request payload
        """
        # Vary parameters for realistic load
        complexity_levels = [0.3, 0.5, 0.7]
        durations = [30, 60, 120]
        budgets = [1.0, 2.0, 5.0]
        
        return {
            "prompt": f"Load test request {request_id}: cinematic scene with dramatic lighting",
            "duration_sec": durations[request_id % len(durations)],
            "complexity": complexity_levels[request_id % len(complexity_levels)],
            "budget_usd": budgets[request_id % len(budgets)]
        }
    
    def _generate_report(self, total_duration: float) -> Dict[str, Any]:
        """
        Generate load test report.
        
        Args:
            total_duration: Total test duration in seconds
            
        Returns:
            Test report dictionary
        """
        # Calculate statistics
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        durations = [r["duration"] for r in self.results]
        costs = [r["cost"] for r in successful]
        qualities = [r["quality"] for r in successful if r["quality"] > 0]
        
        report = {
            "summary": {
                "total_requests": self.num_requests,
                "successful": len(successful),
                "failed": len(failed) + len(self.errors),
                "error_rate": (len(failed) + len(self.errors)) / self.num_requests,
                "total_duration_sec": total_duration,
                "throughput_rps": self.num_requests / total_duration
            },
            "response_time": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "mean": statistics.mean(durations) if durations else 0,
                "median": statistics.median(durations) if durations else 0,
                "p95": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else 0,
                "p99": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else 0
            },
            "cost": {
                "total": sum(costs),
                "mean": statistics.mean(costs) if costs else 0,
                "median": statistics.median(costs) if costs else 0
            },
            "quality": {
                "mean": statistics.mean(qualities) if qualities else 0,
                "median": statistics.median(qualities) if qualities else 0
            },
            "errors": {
                "count": len(self.errors),
                "types": self._categorize_errors()
            }
        }
        
        # Print report
        self._print_report(report)
        
        return report
    
    def _categorize_errors(self) -> Dict[str, int]:
        """
        Categorize errors by type.
        
        Returns:
            Error type counts
        """
        error_types = {}
        
        for error in self.errors:
            error_type = error.get("error", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return error_types
    
    def _print_report(self, report: Dict[str, Any]):
        """
        Print formatted report.
        
        Args:
            report: Test report dictionary
        """
        print(f"\n{'='*60}")
        print("LOAD TEST REPORT")
        print(f"{'='*60}\n")
        
        # Summary
        print("Summary:")
        print(f"  Total Requests: {report['summary']['total_requests']}")
        print(f"  Successful: {report['summary']['successful']}")
        print(f"  Failed: {report['summary']['failed']}")
        print(f"  Error Rate: {report['summary']['error_rate']*100:.2f}%")
        print(f"  Total Duration: {report['summary']['total_duration_sec']:.2f}s")
        print(f"  Throughput: {report['summary']['throughput_rps']:.2f} req/s")
        
        # Response Time
        print(f"\nResponse Time:")
        print(f"  Min: {report['response_time']['min']:.2f}s")
        print(f"  Max: {report['response_time']['max']:.2f}s")
        print(f"  Mean: {report['response_time']['mean']:.2f}s")
        print(f"  Median: {report['response_time']['median']:.2f}s")
        print(f"  P95: {report['response_time']['p95']:.2f}s")
        print(f"  P99: {report['response_time']['p99']:.2f}s")
        
        # Cost
        print(f"\nCost:")
        print(f"  Total: ${report['cost']['total']:.2f}")
        print(f"  Mean: ${report['cost']['mean']:.2f}")
        print(f"  Median: ${report['cost']['median']:.2f}")
        
        # Quality
        print(f"\nQuality:")
        print(f"  Mean Score: {report['quality']['mean']:.2f}/10")
        print(f"  Median Score: {report['quality']['median']:.2f}/10")
        
        # Errors
        if report['errors']['count'] > 0:
            print(f"\nErrors:")
            for error_type, count in report['errors']['types'].items():
                print(f"  {error_type}: {count}")
        
        print(f"\n{'='*60}\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AIPROD API Load Testing")
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="API base URL"
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = LoadTester(
        base_url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency
    )
    
    # Run test
    report = await tester.run()
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
