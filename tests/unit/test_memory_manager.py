# test_memory_manager.py
import pytest
import os
from src.memory.memory_manager import MemoryManager
from src.utils.monitoring import LOG_FILE

def test_set_and_get():
	mm = MemoryManager()
	mm.set('sanitized_input', {'foo': 'bar'})
	assert mm.get('sanitized_input') == {'foo': 'bar'}

def test_export():
	mm = MemoryManager()
	mm.set('sanitized_input', 123)
	mm.set('prompt_bundle', 'abc')
	exported = mm.export()
	assert exported['sanitized_input'] == 123
	assert exported['prompt_bundle'] == 'abc'

def test_clear():
	mm = MemoryManager()
	mm.set('sanitized_input', 1)
	mm.clear()
	assert mm.get('sanitized_input') is None

def test_validate_success():
	mm = MemoryManager()
	mm.set('sanitized_input', 'ok')
	mm.set('prompt_bundle', 'bundle')
	mm.set('optimized_backend_selection', 'backend')
	mm.set('cost_certification', 'cert')
	mm.set('generated_assets', 'assets')
	mm.set('technical_validation_report', 'report')
	mm.set('consistency_report', 'consistency')
	mm.set('final_approval', 'approved')
	mm.set('delivery_manifest', 'manifest')
	assert mm.validate() is True

def test_validate_failure():
	mm = MemoryManager()
	# missing required fields
	assert mm.validate() is False

def test_cache_set_and_get():
    mm = MemoryManager()
    mm.set('foo', 'bar', cache=True)
    assert mm.get('foo', cache=True) == 'bar'


def test_cache_expiry():
    mm = MemoryManager()
    mm.cache.ttl = 1  # 1 seconde pour le test
    mm.set('expire_key', 'baz', cache=True)
    import time
    time.sleep(2)
    assert mm.get('expire_key', cache=True) is None


def test_logging_write_and_read(tmp_path):
	mm = MemoryManager()
	mm.set('log_test', 123)
	mm.get('log_test')
	# Vérifie que le fichier de log existe et contient les entrées
	assert os.path.exists(LOG_FILE)
	with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
		logs = f.read()
	assert 'Write: key=log_test' in logs
	assert 'Read: key=log_test' in logs


def test_validate_data_logging():
	mm = MemoryManager()
	mm.set('complexity_score', 2)  # hors bornes
	assert not mm.validate_data()
	with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
		logs = f.read()
	assert 'Validation data error' in logs
