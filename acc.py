import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache
import asyncio
import aiofiles
import uuid
import time
import random
from io import BytesIO
import base64
import hashlib
import socket
import pickle
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import openai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - [%(threadName)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('data_product_platform.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Performance monitoring decorator
def performance_monitor(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Async performance monitoring decorator
def async_performance_monitor(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"Async function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Enhanced caching decorator
def smart_cache(maxsize=128, ttl=3600):
    cache = {}
    cache_order = []
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = hashlib.md5(f"{str(args)}_{str(kwargs)}".encode()).hexdigest()
            current_time = time.time()
            if key in cache and current_time - cache[key]['timestamp'] < ttl:
                if key in cache_order:
                    cache_order.remove(key)
                cache_order.append(key)
                return cache[key]['value']
            
            result = func(*args, **kwargs)
            cache[key] = {'value': result, 'timestamp': current_time}
            cache_order.append(key)
            
            if len(cache) > maxsize:
                oldest_key = cache_order.pop(0)
                if oldest_key in cache:
                    del cache[oldest_key]
            return result
        return wrapper
    return decorator

@contextmanager
def timing_block(name):
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    logger.debug(f"Block '{name}' executed in {execution_time:.4f} seconds")

# Enhanced Business Analyst Agent
class BusinessAnalystAgent:
    def __init__(self):
        self.business_glossary = {
            "customer": "A person or organization that purchases goods or services",
            "product": "An item offered for sale",
            "transaction": "A record of a purchase or sale",
            "revenue": "Income generated from business activities",
            "inventory": "Goods available for sale",
            "supplier": "A company that provides products or services",
            "location": "A physical place where business is conducted",
            "employee": "A person who works for the organization",
            "department": "An organizational unit within the company",
            "sales": "Activities related to selling products or services",
            "churn": "When customers stop doing business with a company",
            "lifetime value": "Predicted revenue from a customer over their relationship",
            "conversion rate": "Percentage of visitors who take a desired action",
            "acquisition cost": "Cost to acquire a new customer",
            "retention rate": "Percentage of customers who continue using products/services",
            "dashboard": "Visual display of key performance indicators"
        }
        self.synonyms = {
            "customer": ["client", "buyer", "consumer", "patron", "shopper", "user"],
            "product": ["item", "goods", "merchandise", "offering", "solution"],
            "revenue": ["income", "earnings", "sales", "turnover", "proceeds"],
            "churn": ["attrition", "defection", "turnover", "customer loss"],
            "lifetime value": ["CLV", "LTV", "customer lifetime value", "CLTV"],
            "dashboard": ["report", "visualization", "analytics view", "metrics display"]
        }
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_cache = {}
            self.nlp_enabled = True
        except Exception as e:
            logger.warning(f"NLP features partially disabled: {str(e)}")
            self.nlp_enabled = False
            
        try:
            self.openai_client = openai.OpenAI()
            self.llm_enabled = True
        except Exception as e:
            logger.warning(f"LLM features disabled: {str(e)}")
            self.llm_enabled = False

    @lru_cache(maxsize=128)
    @async_performance_monitor
    async def analyze_use_case(self, use_case_text: str) -> Dict:
        logger.info("Analyzing use case text")
        tasks = [
            self._extract_entities(use_case_text),
            self._extract_metrics(use_case_text),
            self._determine_frequency(use_case_text),
            self._identify_aggregations(use_case_text),
            self._extract_filters(use_case_text),
            self._extract_relationships(use_case_text),
            self._extract_user_personas(use_case_text),
            self._extract_business_goals(use_case_text)
        ]
        
        results = await asyncio.gather(*tasks)
        entities, metrics, frequency, aggregations, filters, relationships, personas, goals = results
        
        insights = []
        recommendations = []
        if self.llm_enabled:
            try:
                insights = await self._generate_llm_insights(use_case_text)
                recommendations = await self._generate_recommendations(use_case_text)
            except Exception as e:
                logger.error(f"Error generating LLM outputs: {str(e)}")
                
        return {
            "entities": entities,
            "metrics": metrics,
            "frequency": frequency,
            "aggregations": aggregations,
            "filters": filters,
            "relationships": relationships,
            "personas": personas,
            "business_goals": goals,
            "insights": insights,
            "recommendations": recommendations,
            "original_text": use_case_text,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence_score": self._calculate_confidence_score(entities, metrics, frequency)
        }

    @async_performance_monitor
    async def _extract_entities(self, text: str) -> List[str]:
        entities = set()
        text_lower = text.lower()
        
        for entity, definition in self.business_glossary.items():
            if entity in text_lower:
                entities.add(entity)
            for synonym in self.synonyms.get(entity, []):
                if synonym in text_lower:
                    entities.add(entity)
                    
        if self.nlp_enabled:
            tokens = word_tokenize(text_lower)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
            
            for i in range(len(filtered_tokens) - 1):
                bigram = ' '.join(filtered_tokens[i:i+2])
                for entity in self.business_glossary:
                    if self._semantic_similarity(bigram, entity) > 0.8:
                        entities.add(entity)
                        
            for token in filtered_tokens:
                for entity in self.business_glossary:
                    if self._semantic_similarity(token, entity) > 0.85:
                        entities.add(entity)
        return list(entities)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        if not self.nlp_enabled:
            return 0.0
        if text1 not in self.embeddings_cache:
            self.embeddings_cache[text1] = self.embedding_model.encode([text1])[0]
        if text2 not in self.embeddings_cache:
            self.embeddings_cache[text2] = self.embedding_model.encode([text2])[0]
            
        emb1 = self.embeddings_cache[text1]
        emb2 = self.embeddings_cache[text2]
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    @async_performance_monitor
    async def _extract_metrics(self, text: str) -> List[str]:
        potential_metrics = [
            "count", "sum", "average", "total", "revenue", "profit", "sales",
            "quantity", "volume", "rate", "ratio", "percentage", "frequency",
            "churn", "retention", "conversion", "growth", "cost", "margin"
        ]
        found_metrics = []
        text_lower = text.lower()
        
        for metric in potential_metrics:
            if metric in text_lower:
                found_metrics.append(metric)
                
        matches = re.findall(r"([0-9]+)\s*%\s*(\w+)", text_lower)
        for match in matches:
            if match[1] not in found_metrics and match[1] not in ["of", "to", "from"]:
                found_metrics.append(match[1])
                
        matches = re.findall(r"(kpi|metric|measure)s?\s+(?:like|such as|including)?\s+([a-z\s,]+)", text_lower)
        for match in matches:
            metric_list = match[1].split(',')
            for metric in metric_list:
                clean_metric = metric.strip()
                if clean_metric and clean_metric not in found_metrics:
                    found_metrics.append(clean_metric)
        return found_metrics

    @async_performance_monitor
    async def _determine_frequency(self, text: str) -> str:
        time_periods = {
            "real-time": ["real time", "real-time", "live", "streaming", "continuous", "instant"],
            "hourly": ["hourly", "hour", "every hour"],
            "daily": ["daily", "day", "every day", "per day"],
            "weekly": ["weekly", "week", "every week", "per week"],
            "monthly": ["monthly", "month", "every month", "per month"],
            "quarterly": ["quarterly", "quarter", "every quarter"],
            "yearly": ["yearly", "annual", "year", "every year", "per year"]
        }
        text_lower = text.lower()
        
        for frequency, keywords in time_periods.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return frequency
        return "daily"

    @async_performance_monitor
    async def _identify_aggregations(self, text: str) -> List[str]:
        aggregation_keywords = {
            "sum": ["sum", "total", "aggregate", "combined", "overall"],
            "average": ["average", "avg", "mean", "typical", "expected"],
            "count": ["count", "number of", "quantity", "frequency", "occurrences"],
            "min": ["minimum", "min", "lowest", "smallest", "least"],
            "max": ["maximum", "max", "highest", "largest", "most"]
        }
        found_aggregations = []
        text_lower = text.lower()
        
        for agg, keywords in aggregation_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_aggregations.append(agg)
                    break
        return found_aggregations

    @async_performance_monitor
    async def _extract_filters(self, text: str) -> List[Dict[str, str]]:
        filter_categories = {
            "demographic": ["age", "gender", "income", "education", "location", "region"],
            "time": ["date", "period", "year", "month", "day", "hour"],
            "product": ["category", "type", "brand", "size", "color", "price"],
            "behavior": ["segment", "status", "activity", "usage", "frequency"]
        }
        found_filters = []
        text_lower = text.lower()
        
        filter_patterns = [
            r"filter(?:ed)?\s+by\s+([a-z\s,]+)",
            r"group(?:ed)?\s+by\s+([a-z\s,]+)",
            r"segment(?:ed)?\s+by\s+([a-z\s,]+)"
        ]
        
        for pattern in filter_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                filter_items = match.split(',')
                for item in filter_items:
                    clean_item = item.strip()
                    if clean_item:
                        category = "other"
                        for cat, keywords in filter_categories.items():
                            if any(kw in clean_item for kw in keywords):
                                category = cat
                                break
                        found_filters.append({
                            "name": clean_item,
                            "category": category,
                            "required": "must" in text_lower[:text_lower.find(clean_item)]
                        })
        return found_filters

    @async_performance_monitor
    async def _extract_relationships(self, text: str) -> List[Dict[str, str]]:
        relationships = []
        text_lower = text.lower()
        
        relation_patterns = [
            r"([\w\s]+)\s+(?:related to|linked to|associated with)\s+([\w\s]+)",
            r"([\w\s]+)\s+(?:depends on|based on|derived from)\s+([\w\s]+)"
        ]
        
        for pattern in relation_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                entity1, entity2 = match
                if len(entity1) > 3 and len(entity2) > 3:
                    relationships.append({
                        "source": entity1.strip(),
                        "target": entity2.strip(),
                        "type": "associated"
                    })
        return relationships

    @async_performance_monitor
    async def _extract_user_personas(self, text: str) -> List[Dict[str, str]]:
        personas = []
        text_lower = text.lower()
        
        persona_patterns = [
            r"(?:for|by|to)\s+(?:the)?\s*([\w\s]+(?:team|manager|analyst|executive|user|customer|client|department)s?)"
        ]
        
        for pattern in persona_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                persona = match.strip()
                if len(persona) > 3:
                    role = next((term for term in ["team", "manager", "analyst", "executive", "user"] if term in persona), "user")
                    personas.append({
                        "name": persona,
                        "role": role,
                        "priority": "high" if "key" in text_lower[:text_lower.find(persona)] else "medium"
                    })
        return personas if personas else [{"name": "business users", "role": "user", "priority": "medium"}]

    @async_performance_monitor
    async def _extract_business_goals(self, text: str) -> List[Dict[str, Union[str, int]]]:
        goals = []
        text_lower = text.lower()
        
        goal_patterns = [
            r"(increase|improve|reduce|maximize)\s+([\w\s]+)"
        ]
        
        priority_terms = {"high": ["critical", "essential"], "medium": ["important"], "low": ["nice"]}
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                action, target = match
                priority = "medium"
                context = text_lower[:text_lower.find(action)]
                for p, terms in priority_terms.items():
                    if any(term in context for term in terms):
                        priority = p
                        break
                goals.append({
                    "action": action,
                    "target": target,
                    "priority": priority,
                    "score": {"increase": 5, "improve": 4, "reduce": 5, "maximize": 5}.get(action, 3)
                })
        return sorted(goals, key=lambda x: x["score"], reverse=True)

    @async_performance_monitor
    async def _generate_llm_insights(self, text: str) -> List[Dict[str, str]]:
        if not self.llm_enabled:
            return []
        try:
            prompt = f"Extract key insights from: {text}\nFocus on business problem, stakeholders, metrics, challenges, and opportunities."
            response = self.openai_client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=250,
                temperature=0.7
            )
            insights_text = response.choices[0].text.strip()
            return [{"id": str(i+1), "text": line.strip(), "category": "general", "confidence": 0.8} 
                   for i, line in enumerate(insights_text.split('\n')) if line.strip()]
        except Exception as e:
            logger.error(f"Error generating LLM insights: {str(e)}")
            return []

    @async_performance_monitor
    async def _generate_recommendations(self, text: str) -> List[Dict[str, str]]:
        if not self.llm_enabled:
            return []
        try:
            prompt = f"Based on this use case: {text}\nSuggest additional metrics, filters, or features to enhance the data product."
            response = self.openai_client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            )
            rec_text = response.choices[0].text.strip()
            return [{"id": str(i+1), "text": line.strip(), "category": "enhancement", "confidence": 0.75} 
                   for i, line in enumerate(rec_text.split('\n')) if line.strip()]
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def _calculate_confidence_score(self, entities: List[str], metrics: List[str], frequency: str) -> float:
        score = 0.5
        if entities:
            score += min(0.2, len(entities) * 0.04)
        if metrics:
            score += min(0.2, len(metrics) * 0.04)
        if frequency != "daily":
            score += 0.1
        return min(0.95, score)

# Enhanced Data Designer Agent
class DataDesignerAgent:
    def __init__(self):
        self.common_data_types = {
            "id": "string",
            "name": "string",
            "date": "date",
            "timestamp": "timestamp",
            "amount": "decimal",
            "quantity": "integer",
            "price": "decimal",
            "description": "string",
            "email": "string",
            "phone": "string",
            "address": "string",
            "status": "string",
            "category": "string",
            "url": "string",
            "boolean": "boolean",
            "json": "json",
            "array": "array",
            "uuid": "uuid",
            "geography": "geography"
        }
        self.design_patterns = {
            "transactional": {
                "structure": ["id", "timestamp", "entity_id", "amount", "status", "reference"],
                "storage": "relational",
                "update_pattern": "append",
                "partitioning": "by_date"
            },
            "analytical": {
                "structure": ["dimension_keys", "metrics", "time_dimension", "aggregation_level"],
                "storage": "columnar",
                "update_pattern": "overwrite",
                "partitioning": "by_dimension"
            },
            "master_data": {
                "structure": ["id", "name", "attributes", "metadata", "valid_from", "valid_to"],
                "storage": "key_value",
                "update_pattern": "scd_type2",
                "partitioning": "none"
            },
            "reference_data": {
                "structure": ["code", "description", "category", "valid_from", "valid_to"],
                "storage": "key_value",
                "update_pattern": "scd_type1",
                "partitioning": "none"
            }
        }

    @async_performance_monitor
    async def design_data_product(self, business_requirements: Dict, source_systems: List[Dict]) -> Dict:
        product_type = self._determine_product_type(business_requirements)
        schema = await self._design_schema(business_requirements, product_type, source_systems)
        metadata = self._design_metadata(business_requirements, product_type)
        access_patterns = self._configure_access_patterns(business_requirements, product_type)
        return {
            "product_type": product_type,
            "schema": schema,
            "metadata": metadata,
            "access_patterns": access_patterns,
            "storage_format": self._determine_storage_format(product_type),
            "update_frequency": business_requirements.get("frequency", "daily"),
            "partitioning_strategy": self._determine_partitioning(product_type, business_requirements)
        }

    def _determine_product_type(self, requirements: Dict) -> str:
        text = requirements.get("original_text", "").lower()
        if any(word in text for word in ["report", "dashboard", "analyze", "trend", "kpi"]):
            return "analytical"
        elif any(word in text for word in ["transaction", "process", "record", "event"]):
            return "transactional"
        elif any(word in text for word in ["master data", "reference", "lookup", "code"]):
            return "master_data" if "master" in text else "reference_data"
        return "analytical"

    async def _design_schema(self, requirements: Dict, product_type: str, source_systems: List[Dict]) -> List[Dict]:
        schema = []
        pattern = self.design_patterns[product_type]
        for field_type in pattern["structure"]:
            if field_type == "dimension_keys" and "entities" in requirements:
                for entity in requirements["entities"]:
                    schema.extend([
                        {"name": f"{entity}_id", "type": "string", "description": f"Unique identifier for {entity}", "required": True},
                        {"name": f"{entity}_name", "type": "string", "description": f"Name of the {entity}", "required": False}
                    ])
            elif field_type == "metrics" and "metrics" in requirements:
                for metric in requirements["metrics"]:
                    data_type = "decimal" if metric in ["revenue", "profit", "amount"] else "integer"
                    schema.append({
                        "name": metric,
                        "type": data_type,
                        "description": f"{metric.capitalize()} value",
                        "required": True
                    })
            elif field_type == "time_dimension":
                schema.append({
                    "name": "date",
                    "type": "date",
                    "description": "Date of the record",
                    "required": True
                })
            elif field_type == "aggregation_level" and "aggregations" in requirements:
                schema.append({
                    "name": "aggregation_level",
                    "type": "string",
                    "description": "Level of aggregation",
                    "required": True
                })
            else:
                field_name = field_type
                field_type = self._guess_field_type(field_name)
                schema.append({
                    "name": field_name,
                    "type": field_type,
                    "description": f"{field_name.replace('_', ' ').capitalize()}",
                    "required": field_name in ["id", "timestamp", "entity_id"]
                })
        schema.extend([
            {"name": "created_at", "type": "timestamp", "description": "Record creation time", "required": True},
            {"name": "updated_at", "type": "timestamp", "description": "Last update time", "required": True},
            {"name": "source_system", "type": "string", "description": "Data origin", "required": True}
        ])
        return schema

    def _guess_field_type(self, field_name: str) -> str:
        for key, data_type in self.common_data_types.items():
            if key in field_name.lower():
                return data_type
        return "string"

    def _design_metadata(self, requirements: Dict, product_type: str) -> Dict:
        entities = requirements.get("entities", [])
        entity_string = ", ".join(entities) if entities else "various entities"
        metrics = requirements.get("metrics", [])
        metrics_string = ", ".join(metrics) if metrics else "various metrics"
        description = f"Data product for {entity_string} with {metrics_string}"
        return {
            "name": f"{product_type.capitalize()}_Data_Product_{datetime.now().strftime('%Y%m%d')}",
            "description": description,
            "owner": "Data Team",
            "tags": entities + ["data_product", product_type],
            "version": "1.0.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "update_frequency": requirements.get("frequency", "daily"),
            "retention_period": "3 years",
            "classification": "internal"
        }

    def _configure_access_patterns(self, requirements: Dict, product_type: str) -> List[Dict]:
        patterns = [{"name": "default", "description": "Default query pattern", "type": "query"}]
        if "frequency" in requirements:
            patterns.append({
                "name": "time_based",
                "description": f"{requirements['frequency']} data access",
                "type": "time_filter",
                "time_field": "date" if product_type == "analytical" else "timestamp"
            })
        if "entities" in requirements and requirements["entities"]:
            entity = requirements["entities"][0]
            patterns.append({
                "name": f"by_{entity}",
                "description": f"Filter by {entity}",
                "type": "entity_filter",
                "entity_field": f"{entity}_id"
            })
        return patterns

    def _determine_storage_format(self, product_type: str) -> str:
        storage_map = {
            "analytical": "parquet",
            "transactional": "delta",
            "master_data": "delta",
            "reference_data": "json"
        }
        return storage_map.get(product_type, "parquet")

    def _determine_partitioning(self, product_type: str, requirements: Dict) -> Dict:
        partitioning = self.design_patterns[product_type]["partitioning"]
        if partitioning == "by_date":
            return {"strategy": "by_date", "field": "date", "format": "year/month/day"}
        elif partitioning == "by_dimension" and "entities" in requirements and requirements["entities"]:
            return {"strategy": "by_dimension", "field": f"{requirements['entities'][0]}_id", "format": "hashed_bucket"}
        return {"strategy": "none"}

# Data Source Agent (simplified)
class DataSourceAgent:
    def __init__(self):
        self.source_systems = [
            {
                "name": "CRM",
                "type": "OLTP",
                "tables": [
                    {
                        "name": "customers",
                        "attributes": [
                            {"name": "customer_id", "type": "string"},
                            {"name": "customer_name", "type": "string"}
                        ]
                    }
                ]
            }
        ]

    def identify_source_attributes(self, business_requirements: Dict, target_schema: List[Dict]) -> Dict:
        source_mapping = {}
        for target_attr in target_schema:
            target_name = target_attr["name"]
            source_candidates = self._find_source_candidates(target_name, business_requirements)
            source_mapping[target_name] = source_candidates if source_candidates else {"status": "missing", "sources": []}
        return source_mapping

    def _find_source_candidates(self, target_attribute: str, requirements: Dict) -> Dict:
        candidates = []
        for system in self.source_systems:
            for table in system["tables"]:
                for attr in table["attributes"]:
                    if target_attribute in attr["name"] or attr["name"] in target_attribute:
                        candidates.append({
                            "system": system["name"],
                            "table": table["name"],
                            "attribute": attr["name"],
                            "type": attr["type"],
                            "match_quality": "high"
                        })
        return {"status": "matched", "sources": candidates} if candidates else None

    def get_source_systems(self) -> List[Dict]:
        return self.source_systems

# Data Mapper Agent (simplified)
class DataMapperAgent:
    def generate_mappings(self, target_schema: List[Dict], source_mappings: Dict) -> List[Dict]:
        mappings = []
        for target_attr in target_schema:
            target_name = target_attr["name"]
            mapping_entry = {"target_attribute": target_name, "target_type": target_attr["type"], "required": target_attr.get("required", False)}
            if target_name in source_mappings and source_mappings[target_name]["status"] == "matched":
                best_match = source_mappings[target_name]["sources"][0]
                mapping_entry.update({
                    "mapping_type": "direct",
                    "source_system": best_match["system"],
                    "source_table": best_match["table"],
                    "source_attribute": best_match["attribute"],
                    "source_type": best_match["type"]
                })
            else:
                mapping_entry.update({"mapping_type": "missing", "notes": "No source found"})
            mappings.append(mapping_entry)
        return mappings

# Data Product Certifier (simplified)
class DataProductCertifier:
    def certify_data_product(self, data_product_design: Dict, mappings: List[Dict]) -> Dict:
        certification_results = {
            "data_quality": [],
            "data_governance": []
        }
        
        total_mappings = len(mappings)
        mapped_count = sum(1 for m in mappings if m["mapping_type"] == "direct")
        mapping_completeness = (mapped_count / total_mappings * 100) if total_mappings > 0 else 0
        certification_results["data_quality"].append({
            "standard": "Mapping Completeness",
            "status": "passed" if mapping_completeness >= 80 else "failed",
            "score": mapping_completeness,
            "notes": f"{mapped_count}/{total_mappings} attributes mapped"
        })
        
        required_mapped = sum(1 for m in mappings if m.get("required") and m["mapping_type"] == "direct")
        required_total = sum(1 for m in mappings if m.get("required"))
        required_completeness = (required_mapped / required_total * 100) if required_total > 0 else 100
        certification_results["data_quality"].append({
            "standard": "Required Fields Mapped",
            "status": "passed" if required_completeness == 100 else "failed",
            "score": required_completeness,
            "notes": f"{required_mapped}/{required_total} required fields mapped"
        })
        
        has_owner = "owner" in data_product_design.get("metadata", {})
        certification_results["data_governance"].append({
            "standard": "Ownership Defined",
            "status": "passed" if has_owner else "failed",
            "score": 100 if has_owner else 0,
            "notes": "Owner specified" if has_owner else "No owner specified"
        })
        
        scores = [item["score"] for category in certification_results.values() for item in category]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        data_product_design["certification"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": round(overall_score, 2),
            "results": certification_results
        }
        return data_product_design

    def define_ingress_process(self, data_product: Dict, mappings: List[Dict]) -> Dict:
        return {"type": "batch", "frequency": data_product.get("update_frequency", "daily")}

    def define_egress_process(self, data_product: Dict) -> Dict:
        return {"access_methods": [{"type": "api", "protocol": "https"}]}

# Orchestrator
class DataProductOrchestrator:
    def __init__(self):
        self.business_analyst = BusinessAnalystAgent()
        self.data_designer = DataDesignerAgent()
        self.data_source_agent = DataSourceAgent()
        self.data_mapper = DataMapperAgent()
        self.data_certifier = DataProductCertifier()

    @async_performance_monitor
    async def process_use_case(self, use_case_text: str) -> Dict:
        results = {}
        business_reqs, source_systems = await asyncio.gather(
            self.business_analyst.analyze_use_case(use_case_text),
            asyncio.to_thread(self.data_source_agent.get_source_systems)
        )
        results["business_requirements"] = business_reqs
        results["source_systems"] = source_systems
        
        data_product_design = await self.data_designer.design_data_product(business_reqs, source_systems)
        results["data_product_design"] = data_product_design
        
        source_mappings = self.data_source_agent.identify_source_attributes(business_reqs, data_product_design["schema"])
        results["source_mappings"] = source_mappings
        
        mappings = self.data_mapper.generate_mappings(data_product_design["schema"], source_mappings)
        results["mappings"] = mappings
        
        data_product_design["ingress_process"] = self.data_certifier.define_ingress_process(data_product_design, mappings)
        data_product_design["egress_process"] = self.data_certifier.define_egress_process(data_product_design)
        
        results["certified_data_product"] = self.data_certifier.certify_data_product(data_product_design, mappings)
        return results

    async def serialize_results(self, results: Dict, output_dir: str = "output") -> Dict:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_paths = {}
        for key, data in results.items():
            file_path = os.path.join(output_dir, f"{key}.json")
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            file_paths[key] = file_path
        return file_paths

# PDF Report Generation
def generate_pdf_report(results, customer_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    
    c.drawString(100, y, "Data Product Studio Report")
    y -= 30
    
    c.drawString(100, y, "Use Case:")
    y -= 20
    c.drawString(120, y, results["business_requirements"]["original_text"][:100] + "...")
    y -= 40
    
    if customer_data is not None:
        c.drawString(100, y, "Analytics Summary:")
        y -= 20
        stats = customer_data.describe().to_string().split('\n')[:5]
        for line in stats:
            c.drawString(120, y, line[:80])
            y -= 15
        y -= 20
    
    c.drawString(100, y, "Certification:")
    y -= 20
    cert = results["certified_data_product"]["certification"]
    c.drawString(120, y, f"Overall Score: {cert['overall_score']}%")
    y -= 15
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# User Manager Class for Login
class UserManager:
    def __init__(self):
        if "users" not in st.session_state:
            st.session_state.users = {
                "admin@example.com": {
                    "user_id": "user_1",
                    "email": "admin@example.com",
                    "password": hashlib.sha256("admin123".encode()).hexdigest(),
                    "full_name": "Admin User"
                }
            }
        if "user" not in st.session_state:
            st.session_state.user = None

    def register_user(self, email, password, full_name):
        if email not in st.session_state.users:
            st.session_state.users[email] = {
                "user_id": f"user_{len(st.session_state.users) + 1}",
                "email": email,
                "password": hashlib.sha256(password.encode()).hexdigest(),
                "full_name": full_name
            }
            return True
        return False

    def login_user(self, email, password):
        user = st.session_state.users.get(email)
        if user and user["password"] == hashlib.sha256(password.encode()).hexdigest():
            st.session_state.user = user
            return user
        return None

    def logout_user(self):
        st.session_state.user = None

# UI/UX Enhancement Functions
def custom_spinner(text="Processing..."):
    spinner_html = f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
        <div style="width: 24px; height: 24px; border: 3px solid #E8ECEF; border-top: 3px solid #4285F4; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        <span style="margin-left: 10px; color: #5F6368; font-size: 14px;">{text}</span>
    </div>
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    return st.markdown(spinner_html, unsafe_allow_html=True)

def run_streamlit_app():
    st.set_page_config(page_title="Data Product Studio", layout="wide", page_icon="🔍")
    
    # Minimalistic Google-Inspired CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    :root {
        --primary: #4285F4;  /* Google Blue */
        --background: #FFFFFF;
        --text: #202124;
        --text-secondary: #5F6368;
        --border: #DADCE0;
        --card-bg: #F8F9FA;
        --hover: #E8ECEF;
    }
    body, .main {
        background: var(--background);
        color: var(--text);
        font-family: 'Roboto', sans-serif;
        font-weight: 400;
        font-size: 14px;
        line-height: 1.5;
        margin: 0;
        padding: 0;
    }
    h1 {
        font-family: 'Roboto', sans-serif;
        font-size: 24px;
        font-weight: 500;
        color: var(--text);
        margin-bottom: 20px;
    }
    h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        font-size: 18px;
        color: var(--text);
        margin-bottom: 15px;
    }
    .sidebar .sidebar-content {
        background: var(--background);
        border-right: 1px solid var(--border);
        padding: 20px;
    }
    .stButton>button {
        background: var(--primary);
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 500;
        transition: background 0.2s ease;
    }
    .stButton>button:hover {
        background: #3267D6;
    }
    .card {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
    }
    .stTextInput, .stTextArea, .stMultiSelect {
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 8px;
        background: #2D3033;
        color: var(--text);
    }
    .stTextInput:hover, .stTextArea:hover, .stMultiSelect:hover {
        border-color: var(--primary);
    }
    .stTabs [role="tab"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-size: 14px;
        color: var(--text-secondary);
    }
    .stTabs [role="tab"]:hover {
        background: var(--hover);
    }
    .stTabs [aria-selected="true"] {
        background: var(--background);
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 4px;
        overflow: hidden;
    }
    .stProgress .st-bo {
        background: var(--hover);
    }
    .stProgress .st-bo > div {
        background: var(--primary);
    }
    </style>
    """, unsafe_allow_html=True)

    user_manager = UserManager()
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'customer_data' not in st.session_state:
        st.session_state.customer_data = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "Login"

    if not st.session_state.user and st.session_state.current_view != "Login":
        st.session_state.current_view = "Login"

    if st.session_state.current_view == "Login":
        st.markdown("<h1>Data Product Studio</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            tab1, tab2 = st.tabs(["Login", "Register"])
            with tab1:
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Sign In", key="login_btn"):
                    user = user_manager.login_user(email, password)
                    if user:
                        st.success(f"Welcome, {user['full_name']}!")
                        st.session_state.current_view = "Home"
                        st.rerun()
            with tab2:
                reg_email = st.text_input("Email", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                full_name = st.text_input("Full Name", key="reg_full_name")
                if st.button("Register", key="reg_btn"):
                    if user_manager.register_user(reg_email, reg_password, full_name):
                        st.success("Registered successfully! Please sign in.")

    else:
        with st.sidebar:
            st.markdown(f"<p style='color: var(--text-secondary);'>{st.session_state.user['full_name']}</p>", unsafe_allow_html=True)
            view_options = ["Home", "Use Case", "Analytics", "Design", "Mappings", "Certification", "Sign Out"]
            st.session_state.current_view = st.radio("", view_options, key="nav_radio")

        if st.session_state.current_view == "Home":
            st.markdown("<h1>Data Product Studio</h1>", unsafe_allow_html=True)
            st.write("Build and analyze data products with ease.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Users", "18,542", "9.3%")
            with col2:
                st.metric("Revenue", "$1.2M", "12.1%")
            with col3:
                st.metric("Retention", "87.6%", "5.2%")

        elif st.session_state.current_view == "Use Case":
            st.markdown("<h1>Use Case</h1>", unsafe_allow_html=True)
            use_case_text = st.text_area("Enter your use case", height=150, key="use_case_input")
            if st.button("Analyze"):
                with st.spinner():
                    custom_spinner("Analyzing...")
                    async def process():
                        orchestrator = DataProductOrchestrator()
                        st.session_state.results = await orchestrator.process_use_case(use_case_text)
                    asyncio.run(process())
                    st.success("Analysis completed")
            
            if st.session_state.results:
                st.markdown("<h2>Results</h2>", unsafe_allow_html=True)
                st.write(f"**Text:** {st.session_state.results['business_requirements']['original_text']}")
                st.write(f"**Entities:** {', '.join(st.session_state.results['business_requirements']['entities'])}")
                st.write(f"**Metrics:** {', '.join(st.session_state.results['business_requirements']['metrics'])}")
                st.write(f"**Frequency:** {st.session_state.results['business_requirements']['frequency']}")

        elif st.session_state.current_view == "Analytics":
            st.markdown("<h1>Analytics</h1>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                st.session_state.customer_data = pd.read_csv(uploaded_file)
            elif st.session_state.customer_data is None:
                customer_data = pd.DataFrame({
                    'customer_id': range(10000),
                    'age': np.random.normal(40, 10, 10000).astype(int),
                    'revenue': np.random.lognormal(5, 1, 10000),
                    'purchases': np.random.poisson(3, 10000),
                    'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
                    'last_purchase': pd.date_range(start='2023-01-01', periods=10000, freq='H'),
                    'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 10000),
                    'satisfaction': np.random.uniform(1, 5, 10000),
                    'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 10000)
                })
                st.session_state.customer_data = customer_data

            data = st.session_state.customer_data.copy()
            col1, col2, col3 = st.columns(3)
            with col1:
                regions = st.multiselect("Regions", options=data['region'].unique(), default=data['region'].unique())
            with col2:
                segments = st.multiselect("Segments", options=data['segment'].unique(), default=data['segment'].unique())
            with col3:
                if 'category' in data.columns:
                    categories = st.multiselect("Categories", options=data['category'].unique(), default=data['category'].unique())
                else:
                    categories = None
            
            filtered_data = data[data['region'].isin(regions) & data['segment'].isin(segments)]
            if categories:
                filtered_data = filtered_data[filtered_data['category'].isin(categories)]
            
            if st.button("Update Data"):
                filtered_data['revenue'] += np.random.normal(0, 10, len(filtered_data))
                st.session_state.customer_data = filtered_data
            
            tabs = st.tabs(["Demographics", "Behavior", "Trends", "Network", "Clusters", "Forecast"])
            
            with tabs[0]:
                fig = px.histogram(filtered_data, x="age", color="segment", title="Age Distribution")
                fig.update_traces(marker=dict(line=dict(width=1, color="#DADCE0")), opacity=0.8)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                fig = px.scatter(filtered_data, x="purchases", y="revenue", color="region", size="satisfaction", title="Purchases vs Revenue")
                fig.update_traces(marker=dict(line=dict(width=1, color="#DADCE0")), opacity=0.8)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                trend_data = filtered_data.groupby(filtered_data['last_purchase'].dt.date).agg({'revenue': 'sum'}).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_data['last_purchase'], y=trend_data['revenue'], name="Revenue", line=dict(color="#4285F4")))
                fig.update_layout(title="Revenue Trends", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[3]:
                G = nx.Graph()
                for _, row in filtered_data.sample(100).iterrows():
                    G.add_edge(row['customer_id'], row['region'], weight=row['revenue'])
                pos = nx.spring_layout(G)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#DADCE0')))
                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='#4285F4')))
                fig.update_layout(title="Network", showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[4]:
                features = filtered_data[['age', 'revenue', 'purchases', 'satisfaction']]
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(scaled_features)
                filtered_data['cluster'] = clusters
                fig = px.scatter(filtered_data, x="purchases", y="revenue", color="cluster", size="satisfaction", title="Clusters")
                fig.update_traces(marker=dict(line=dict(width=1, color="#DADCE0")), opacity=0.8)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(filtered_data.groupby('cluster').agg({'age': 'mean', 'revenue': 'mean', 'purchases': 'mean'}).round(2))
            
            with tabs[5]:
                trend_data = filtered_data.groupby(filtered_data['last_purchase'].dt.date).agg({'revenue': 'sum'}).reset_index()
                trend_data.index = pd.to_datetime(trend_data['last_purchase'])
                model = sm.tsa.ARIMA(trend_data['revenue'], order=(1, 1, 1))
                results = model.fit()
                forecast = results.forecast(steps=30)
                forecast_dates = pd.date_range(start=trend_data['last_purchase'].max(), periods=30, freq='D')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=trend_data['last_purchase'], y=trend_data['revenue'], name="Historical", line=dict(color="#4285F4")))
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="Forecast", line=dict(color="#4285F4", dash='dash')))
                fig.update_layout(title="Forecast", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        elif st.session_state.current_view == "Design" and st.session_state.results:
            st.markdown("<h1>Design</h1>", unsafe_allow_html=True)
            design = st.session_state.results["data_product_design"]
            st.write(f"**Type:** {design.get('product_type')}")
            st.write(f"**Format:** {design.get('storage_format')}")
            st.write(f"**Frequency:** {design.get('update_frequency')}")
            st.markdown("<h2>Schema</h2>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(design.get("schema", [])))

        elif st.session_state.current_view == "Mappings" and st.session_state.results:
            st.markdown("<h1>Mappings</h1>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state.results["mappings"]))

        elif st.session_state.current_view == "Certification" and st.session_state.results:
            st.markdown("<h1>Certification</h1>", unsafe_allow_html=True)
            certification = st.session_state.results["certified_data_product"]["certification"]
            st.progress(int(certification["overall_score"]), text=f"Score: {certification['overall_score']}%")
            st.markdown("<h2>Details</h2>", unsafe_allow_html=True)
            for category, checks in certification["results"].items():
                st.write(f"**{category.replace('_', ' ').title()}**")
                for check in checks:
                    st.write(f"- {check['standard']}: {check['score']}% ({check['notes']})")

        elif st.session_state.current_view == "Sign Out":
            user_manager.logout_user()
            st.session_state.current_view = "Login"
            st.rerun()

        if st.session_state.results:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Export JSON"):
                    async def export_json():
                        orchestrator = DataProductOrchestrator()
                        file_paths = await orchestrator.serialize_results(st.session_state.results)
                        for name, path in file_paths.items():
                            with open(path, 'rb') as f:
                                st.sidebar.download_button(f"{name}.json", f.read(), f"{name}.json")
                    asyncio.run(export_json())
            with col2:
                if st.button("Export PDF"):
                    pdf_buffer = generate_pdf_report(st.session_state.results, st.session_state.customer_data)
                    st.sidebar.download_button("Report.pdf", pdf_buffer, "Data_Product_Studio_Report.pdf")

if __name__ == "__main__":
    run_streamlit_app()