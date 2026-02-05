"""
Advanced query filtering system for AIPROD V33
Supports complex filters, operators, and performance optimization
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime


class FilterOperator(Enum):
    """Supported filter operators"""
    EQ = "="          
    NE = "!="        
    GT = ">"         
    GTE = ">="        
    LT = "<"         
    LTE = "<="        
    IN = "in"         
    NOT_IN = "!in"    
    LIKE = "like"     
    NOT_LIKE = "!like"  
    STARTS = "starts"   
    ENDS = "ends"      


@dataclass
class FilterClause:
    """Represents a single filter clause"""
    field: str
    operator: FilterOperator
    value: Any
    
    def __repr__(self) -> str:
        return f"{self.field} {self.operator.value} {self.value}"


class QueryFilter:
    """Parser and executor for advanced query filters"""
    
    VALID_FIELDS = {
        'status', 'created_at', 'updated_at', 'user_id', 'job_type',
        'priority', 'estimated_cost', 'actual_cost', 'region', 'service',
        'environment', 'name', 'email', 'role', 'enabled', 'cpu', 'memory',
        'disk', 'duration', 'error_count', 'success_count', 'retry_count'
    }
    
    NUMERIC_FIELDS = {'cpu', 'memory', 'disk', 'duration', 'error_count', 
                      'success_count', 'retry_count', 'estimated_cost', 'actual_cost'}
    
    DATE_FIELDS = {'created_at', 'updated_at'}
    
    def __init__(self):
        """Initialize query filter"""
        self.clauses: List[FilterClause] = []
    
    def parse(self, filter_string: str) -> List[FilterClause]:
        """
        Parse filter string into clauses
        Format: field1:operator:value1,field2:operator:value2
        Examples:
            - status:=:completed
            - created_at:>:2026-02-01
            - cpu:>:2
            - name:like:Job
            - status:in:completed|running
        """
        if not filter_string:
            return []
        
        self.clauses = []
        filter_parts = filter_string.split(',')
        
        for part in filter_parts:
            part = part.strip()
            if not part:
                continue
            
            # Parse: field:operator:value with careful operator matching
            clause = self._parse_single_filter(part)
            if clause:
                self.clauses.append(clause)
        
        return self.clauses
    
    def _parse_single_filter(self, filter_str: str) -> Optional[FilterClause]:
        """Parse single filter clause"""
        # Try matching longer operators first to avoid partial matches
        operators_ordered = [
            ('!in', FilterOperator.NOT_IN), ('!like', FilterOperator.NOT_LIKE),
            ('<=', FilterOperator.LTE), ('>=', FilterOperator.GTE),
            ('!=', FilterOperator.NE), ('in', FilterOperator.IN),
            ('like', FilterOperator.LIKE), ('starts', FilterOperator.STARTS),
            ('ends', FilterOperator.ENDS), ('<', FilterOperator.LT),
            ('>', FilterOperator.GT), ('=', FilterOperator.EQ)
        ]
        
        for op_str, op_enum in operators_ordered:
            # Try to find field:operator:value pattern
            pattern = r'^(\w+):' + re.escape(op_str) + r':(.+)$'
            match = re.match(pattern, filter_str)
            if match:
                field, value = match.groups()
                
                if field not in self.VALID_FIELDS:
                    raise ValueError(f"Unknown field: {field}")
                
                parsed_value = self._parse_value(field, value, op_enum)
                return FilterClause(field=field, operator=op_enum, value=parsed_value)
        
        raise ValueError(f"Invalid filter format: {filter_str}")
    
    def _parse_value(self, field: str, value: str, operator: FilterOperator) -> Any:
        """Parse value based on field and operator type"""
        
        # Handle IN/NOT_IN - split by pipe
        if operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            return [v.strip() for v in value.split('|')]
        
        # Parse dates
        if field in self.DATE_FIELDS:
            try:
                if 'T' in value:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                return datetime.strptime(value, '%Y-%m-%d')
            except:
                raise ValueError(f"Invalid date format: {value}")
        
        # Parse numbers for numeric fields
        if field in self.NUMERIC_FIELDS and operator not in [FilterOperator.LIKE, FilterOperator.NOT_LIKE]:
            try:
                return float(value) if '.' in value else int(value)
            except ValueError:
                raise ValueError(f"Invalid number: {value}")
        
        return value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of filters"""
        return {
            'total_filters': len(self.clauses),
            'filters': [str(c) for c in self.clauses],
            'fields': [c.field for c in self.clauses],
            'operators': [c.operator.value for c in self.clauses]
        }
    
    def to_sql_filter(self) -> Dict[str, Any]:
        """Convert to SQL filter format"""
        return {
            'filters': [
                {'field': c.field, 'operator': c.operator.value, 'value': c.value}
                for c in self.clauses
            ]
        }


class FilterExecutor:
    """Executes filters on in-memory data"""
    
    def __init__(self, filter_obj: QueryFilter):
        self.filter_obj = filter_obj
    
    def filter_list(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all filters to list of items"""
        if not self.filter_obj.clauses:
            return items
        
        result = items
        for clause in self.filter_obj.clauses:
            result = self._apply_clause(result, clause)
        
        return result
    
    def _apply_clause(self, items: List[Dict[str, Any]], clause: FilterClause) -> List[Dict[str, Any]]:
        """Apply single clause to items"""
        return [item for item in items if clause.field in item and self._matches(item[clause.field], clause)]
    
    def _matches(self, item_value: Any, clause: FilterClause) -> bool:
        """Check if value matches clause"""
        op = clause.operator
        val = clause.value
        
        if op == FilterOperator.EQ:
            return item_value == val
        elif op == FilterOperator.NE:
            return item_value != val
        elif op == FilterOperator.GT:
            return item_value > val
        elif op == FilterOperator.GTE:
            return item_value >= val
        elif op == FilterOperator.LT:
            return item_value < val
        elif op == FilterOperator.LTE:
            return item_value <= val
        elif op == FilterOperator.IN:
            return item_value in val
        elif op == FilterOperator.NOT_IN:
            return item_value not in val
        elif op == FilterOperator.LIKE:
            return str(val).lower() in str(item_value).lower()
        elif op == FilterOperator.NOT_LIKE:
            return str(val).lower() not in str(item_value).lower()
        elif op == FilterOperator.STARTS:
            return str(item_value).startswith(str(val))
        elif op == FilterOperator.ENDS:
            return str(item_value).endswith(str(val))
        
        return False




class FilterIndexBuilder:
    """Simple index builder for performance optimization"""
    
    def __init__(self):
        self.indexes: Dict[str, Dict[Any, Set[int]]] = {}
    
    def build_index(self, items: List[Dict[str, Any]], field: str) -> Dict[Any, Set[int]]:
        """Build index for a field"""
        index = {}
        for idx, item in enumerate(items):
            if field in item:
                value = item[field]
                if value not in index:
                    index[value] = set()
                index[value].add(idx)
        
        self.indexes[field] = index
        return index
    
    def get_index(self, field: str) -> Optional[Dict[Any, Set[int]]]:
        """Get index for field"""
        return self.indexes.get(field)
    
    def get_matching_indices(self, field: str, clause: FilterClause) -> Optional[Set[int]]:
        """Get matching indices for clause"""
        index = self.get_index(field)
        if not index:
            return None
        
        if clause.operator == FilterOperator.EQ:
            return index.get(clause.value, set())
        elif clause.operator == FilterOperator.IN:
            result = set()
            for v in clause.value:
                result.update(index.get(v, set()))
            return result
        
        return None
    
    def clear(self):
        """Clear all indexes"""
        self.indexes.clear()


# Convenience functions

def parse_filters(filter_string: str) -> QueryFilter:
    """Parse filter string into QueryFilter"""
    qf = QueryFilter()
    qf.parse(filter_string)
    return qf


def apply_filters(items: List[Dict[str, Any]], filter_string: str) -> List[Dict[str, Any]]:
    """Apply filters to items conveniently"""
    if not filter_string:
        return items
    
    qf = parse_filters(filter_string)
    executor = FilterExecutor(qf)
    return executor.filter_list(items)
