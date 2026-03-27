import logging
from typing import Dict, List, Union

from projectdavid.clients.vectors import VectorStoreClient
from projectdavid_common import ValidationInterface
from qdrant_client.http import models


class VectorSearchHandler:

    def __init__(self, assistant_id: str):
        self.assistant_id = assistant_id
        self.vector_store_client = VectorStoreClient()
        self.source_mapping = self._build_dynamic_source_mapping()

    def _build_dynamic_source_mapping(self) -> Dict[str, str]:
        """Dynamically map source types to collection names"""
        stores = self.vector_store_client.get_vector_stores_for_assistant(
            assistant_id=self.assistant_id
        )
        mapping = {}
        for store in stores:
            try:
                _, source_type = store.name.split("-", 1)
                mapping[source_type] = store.collection_name
            except ValueError:
                logging.warning(
                    f"Skipping store '{store.name}' - invalid name format. Expected '{self.assistant_id}-<source_type>'"
                )
        return mapping

    def execute_search(
        self, **kwargs
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        try:
            self._validate_search_params(kwargs)
            handler = getattr(self, f"handle_{kwargs['search_type']}", None)
            if not handler:
                raise ValueError(f"Invalid search_type: {kwargs['search_type']}")
            return handler(kwargs)
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return [
                ValidationInterface.VectorStoreSearchResult(
                    text=f"Search Error: {str(e)}",
                    metadata={"error": True, "type": type(e).__name__},
                    score=0.0,
                    vector_id="",
                    store_id="",
                )
            ]

    def handle_basic_semantic(
        self, params: Dict
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        """Basic vector similarity search"""
        return self.vector_store_client.search_vector_store(
            query_text=params["query"], top_k=params.get("top_k", 5)
        )

    def handle_filtered(
        self, params: Dict
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        """Metadata-filtered search"""
        return self.vector_store_client.search_vector_store(
            query_text=params["query"], filters=params.get("filters", {})
        )

    def handle_complex_filters(
        self, params: Dict
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        """Handle complex filters search with all required parameters"""
        return self.vector_store_client.search_vector_store(
            query_text=params["query"],
            vector_store_id="placeholder",
            filters=params.get("filters", {}),
            top_k=params.get("top_k", 5),
        )

    def _validate_search_params(self, params: Dict):
        """Ensure assistant-provided parameters match tool schema"""
        required = {"query", "search_type", "source_type"}
        if missing := (required - params.keys()):
            raise ValueError(f"Missing required params: {missing}")
        if params["search_type"] not in [
            "basic_semantic",
            "filtered",
            "complex_filters",
            "temporal",
            "explainable",
            "hybrid",
        ]:
            raise ValueError(f"Invalid search_type: {params['search_type']}")
        if params["source_type"] not in ["chat", "documents", "memory"]:
            raise ValueError(f"Invalid source_type: {params['source_type']}")

    def validate_filters(self, filters: dict):
        """Ensure all filter values are JSON-serializable"""
        allowed_types = (str, int, float, bool, type(None))

        def check_value(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if not isinstance(k, str):
                        raise ValueError(f"Invalid filter key type: {type(k)}")
                    check_value(v)
            elif isinstance(value, list):
                for item in value:
                    check_value(item)
            elif not isinstance(value, allowed_types):
                raise ValueError(f"Unserializable type {type(value)} in filters")

        check_value(filters)

    def handle_temporal(
        self, params: Dict
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        """Time-weighted search with validation"""
        self._validate_search_params(params)
        return self.vector_store_client.search_vector_store(
            query_text=params["query"], filters=params.get("filters", {})
        )

    def handle_explainable(
        self, params: Dict
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        """Search with scoring explanations"""
        return self.vector_store_client.search_vector_store(query_text=params["query"])

    def handle_hybrid(
        self, params: Dict
    ) -> List[ValidationInterface.VectorStoreSearchResult]:
        return self.vector_store_service.search_vector_store(
            store_name=self._get_collection_name(params["source_type"]),
            query_text=params["query"],
            search_type="hybrid",
            filters=params.get("filters", {}),
            score_boosts=params.get("score_boosts", {}),
            explain=params.get("explain", False),
        )

    def _get_collection_name(self, source_type: str) -> str:
        """Get validated collection name for source type"""
        collection_name = self.source_mapping.get(source_type)
        if not collection_name:
            available = list(self.source_mapping.keys())
            raise ValueError(
                f"No store found for source_type '{source_type}'. Available types: {available}"
            )
        return collection_name

    def _parse_complex_filters(self, filters: Dict) -> models.Filter:
        return self.vector_store_client._parse_advanced_filters(filters)

    def _recursive_filter_builder(
        self, condition: Union[Dict, List]
    ) -> Union[models.Filter, models.FieldCondition]:
        """Recursively build Qdrant filter conditions"""
        if isinstance(condition, dict):
            for operator in ["$and", "$or", "$not"]:
                if operator in condition:
                    return self._handle_logical_operator(operator, condition[operator])
            return self._build_field_conditions(condition)
        elif isinstance(condition, list):
            return models.Filter(
                must=[self._recursive_filter_builder(c) for c in condition]
            )
        raise ValueError(f"Unsupported condition type: {type(condition)}")

    def _handle_logical_operator(
        self, operator: str, conditions: List
    ) -> models.Filter:
        """Process logical operators"""
        parsed_conditions = [self._recursive_filter_builder(c) for c in conditions]
        if operator == "$and":
            return models.Filter(must=parsed_conditions)
        elif operator == "$or":
            return models.Filter(should=parsed_conditions)
        elif operator == "$not":
            return models.Filter(must_not=parsed_conditions)
        raise ValueError(f"Unknown operator: {operator}")

    def _build_field_conditions(self, field_conditions: Dict) -> models.FieldCondition:
        """Build field-level conditions"""
        for field, condition in field_conditions.items():
            if isinstance(condition, dict):
                return self._parse_comparison_operators(field, condition)
            return models.FieldCondition(
                key=f"metadata.{field}", match=models.MatchValue(value=condition)
            )

    def _parse_comparison_operators(
        self, field: str, operators: Dict
    ) -> models.FieldCondition:
        """Process comparison operators"""
        range_params = {}
        match_params = {}
        for op, value in operators.items():
            if op == "$gt":
                range_params["gt"] = value
            elif op == "$gte":
                range_params["gte"] = value
            elif op == "$lt":
                range_params["lt"] = value
            elif op == "$lte":
                range_params["lte"] = value
            elif op == "$ne":
                match_params["except"] = [value]
            elif op == "$in":
                match_params["any"] = value
            else:
                raise ValueError(f"Unsupported operator: {op}")
        if range_params:
            return models.FieldCondition(
                key=f"metadata.{field}", range=models.Range(**range_params)
            )
        elif match_params:
            if "except" in match_params:
                return models.FieldCondition(
                    key=f"metadata.{field}", match=models.MatchExcept(**match_params)
                )
            return models.FieldCondition(
                key=f"metadata.{field}", match=models.MatchAny(**match_params)
            )
        raise ValueError("No valid operators found")
