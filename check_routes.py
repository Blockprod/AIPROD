from src.api.main import app

routes = [r.path for r in app.routes if hasattr(r, 'path')]
print(f'Total API Routes: {len(routes)}')

deployment_routes = [r for r in routes if 'deployment' in r]
print(f'Deployment Routes: {len(deployment_routes)}')

performance_routes = [r for r in routes if 'performance' in r]
print(f'Performance Routes: {len(performance_routes)}')

monitoring_routes = [r for r in routes if 'monitoring' in r]
print(f'Monitoring Routes: {len(monitoring_routes)}')

print("\nBreakdown:")
print(f"  - Deployment (Multi-region): {len(deployment_routes)}")
print(f"  - Performance (Optimization): {len(performance_routes)}")
print(f"  - Monitoring (Analytics): {len(monitoring_routes)}")
print(f"  - Other routes: {len(routes) - len(deployment_routes) - len(performance_routes) - len(monitoring_routes)}")
