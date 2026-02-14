# Tests - aiprod-core

Unit tests et integration tests pour le module aiprod-core.

## 📁 Structure

```
tests/
├── unit/                    # Tests unitaires
│   ├── test_tools.py
│   ├── test_types.py
│   ├── test_utils.py
│   └── test_loaders.py
├── integration/             # Tests d'intégration
│   ├── test_components.py
│   ├── test_conditioning.py
│   └── test_guidance.py
├── fixtures/                # Données de test
│   ├── sample_data.py
│   └── mock_models.py
└── conftest.py              # Configuration pytest
```

## 🧪 Exécuter les tests

### Tous les tests
```bash
pytest tests/ -v
```

### Tests unitaires seulement
```bash
pytest tests/unit/ -v
```

### Tests d'intégration
```bash
pytest tests/integration/ -v
```

### Avec couverture
```bash
pytest tests/ --cov=aiprod_core --cov-report=html
```

### Mode watch (re-run on change)
```bash
pytest-watch tests/
```

## 📊 Objectifs de couverture

- **Global**: ≥80%
- **aiprod_core/api**: ≥90%
- **aiprod_core/ml**: ≥75%
- **aiprod_core/utils**: ≥85%

## ✅ Checklist avant commit

- [ ] `pytest tests/ -v` passe
- [ ] Couverture ≥80% (`pytest --cov`)
- [ ] Pas de warnings (`pytest -W error`)
- [ ] Lint clean (`flake8 tests/`)
- [ ] Types OK (`mypy tests/`)

## 🔍 Patterns courants

### Mock des modèles
```python
from tests.fixtures import MockModel

@pytest.fixture
def model():
    return MockModel()
```

### Fixtures d'intégration
```python
@pytest.fixture
def sample_data():
    return load_fixture("sample_data.json")
```

### Async tests
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_func()
    assert result is not None
```

---

*Created: 2026-02-10*
