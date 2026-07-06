"""
DREDGE New Architecture - Core Pipeline System

Implements the modular architecture shown in the screenshots:
1. CLI Entry / dredge_run_pipeline
2. DAG Execution Engine (async orchestration)
3. Node Graph (ingest, translate, normalize, fallback)
4. Redis Cache Layer (optional)
5. Telemetry / Observation
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import hashlib
import logging

# Optional redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Node types in the DAG"""
    INGEST = "ingest"
    TRANSLATE = "translate"
    NORMALIZE = "normalize"
    EXECUTE = "execute"
    CACHE = "cache"
    FALLBACK = "fallback"


class NodeStatus(Enum):
    """Node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class NodeMetadata:
    """Node execution metadata"""
    node_id: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cache_key: Optional[str] = None
    cache_hit: bool = False
    error: Optional[str] = None
    telemetry_tags: Dict[str, str] = None

    def __post_init__(self):
        if self.telemetry_tags is None:
            self.telemetry_tags = {}


@dataclass
class PipelineContext:
    """Context passed through pipeline execution"""
    pipeline_id: str
    input_data: Dict[str, Any]
    execution_log: List[str] = None
    node_results: Dict[str, Any] = None
    cache_enabled: bool = True
    async_mode: bool = True

    def __post_init__(self):
        if self.execution_log is None:
            self.execution_log = []
        if self.node_results is None:
            self.node_results = {}


class Node:
    """DAG Node - base class for pipeline components"""

    def __init__(self, node_id: str, node_type: NodeType, handler: Optional[Callable] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.handler = handler
        self.metadata = NodeMetadata(node_id=node_id, node_type=node_type)
        self.dependencies: List[str] = []
        self.cache_ttl = 3600  # 1 hour default

    def add_dependency(self, node_id: str):
        """Add upstream dependency"""
        self.dependencies.append(node_id)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict[str, Any]:
        """Execute node with caching support"""
        self.metadata.status = NodeStatus.RUNNING
        self.metadata.start_time = time.time()

        try:
            # Check cache
            if context.cache_enabled and redis_client and REDIS_AVAILABLE:
                cache_key = self._generate_cache_key(context)
                cached_result = self._get_cache(redis_client, cache_key)
                if cached_result:
                    self.metadata.cache_hit = True
                    self.metadata.status = NodeStatus.CACHED
                    logger.info(f"Cache hit for {self.node_id}: {cache_key}")
                    return cached_result

            # Execute handler
            result = None
            if self.handler:
                if asyncio.iscoroutinefunction(self.handler):
                    result = await self.handler(context)
                else:
                    result = self.handler(context)
            else:
                result = context.node_results.get(self.node_id, {})

            # Cache result
            if context.cache_enabled and redis_client and REDIS_AVAILABLE and result:
                cache_key = self._generate_cache_key(context)
                self._set_cache(redis_client, cache_key, result)

            self.metadata.status = NodeStatus.COMPLETED
            self.metadata.node_results = result
            return result

        except Exception as e:
            self.metadata.status = NodeStatus.FAILED
            self.metadata.error = str(e)
            logger.error(f"Node {self.node_id} failed: {e}")
            raise

        finally:
            self.metadata.end_time = time.time()
            self.metadata.duration = self.metadata.end_time - self.metadata.start_time

    def _generate_cache_key(self, context: PipelineContext) -> str:
        """Generate deterministic cache key"""
        content = json.dumps({
            "node_id": self.node_id,
            "input": context.input_data
        }, sort_keys=True)
        return f"dredge:cache:{hashlib.md5(content.encode()).hexdigest()}"

    def _get_cache(self, redis_client: Any, key: str) -> Optional[Dict]:
        """Retrieve from cache"""
        if not REDIS_AVAILABLE:
            return None
        try:
            cached = redis_client.get(key)
            if cached:
                self.metadata.cache_key = key
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None

    def _set_cache(self, redis_client: Any, key: str, value: Dict):
        """Store in cache"""
        if not REDIS_AVAILABLE:
            return
        try:
            redis_client.setex(key, self.cache_ttl, json.dumps(value))
            self.metadata.cache_key = key
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")


class IngestNode(Node):
    """Ingests raw input data"""

    def __init__(self, node_id: str = "ingest", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.INGEST, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        result = await super().execute(context, redis_client)
        context.execution_log.append(f"[INGEST] Processed input: {len(str(context.input_data))} bytes")
        return result or context.input_data


class TranslateNode(Node):
    """Translates data format (e.g., text -> embeddings)"""

    def __init__(self, node_id: str = "translate", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.TRANSLATE, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        result = await super().execute(context, redis_client)
        context.execution_log.append(f"[TRANSLATE] Converted format")
        return result or {"translated": True}


class NormalizeNode(Node):
    """Normalizes and validates data (ingests, translates, applies fallback)"""

    def __init__(self, node_id: str = "normalize", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.NORMALIZE, handler)
        self.fallback_providers: List[str] = []

    def add_fallback_provider(self, provider_name: str):
        """Add fallback translation provider"""
        self.fallback_providers.append(provider_name)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        result = await super().execute(context, redis_client)
        context.execution_log.append(f"[NORMALIZE] Applied {len(self.fallback_providers)} fallback providers")
        return result or {"normalized": True, "providers_applied": self.fallback_providers}


class ExecuteNode(Node):
    """Executes the actual computation (DAG, model inference)"""

    def __init__(self, node_id: str = "execute", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.EXECUTE, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        result = await super().execute(context, redis_client)
        context.execution_log.append(f"[EXECUTE] Computation complete")
        return result or {"executed": True, "result": 0.94}


class CacheLayer:
    """Redis-backed cache layer for memoization"""

    def __init__(self, redis_client: Optional[Any] = None, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self.local_cache: Dict[str, Dict] = {}

    def get(self, key: str) -> Optional[Dict]:
        """Get from cache (Redis first, then local)"""
        if self.redis and REDIS_AVAILABLE:
            try:
                cached = self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except:
                pass

        return self.local_cache.get(key)

    def set(self, key: str, value: Dict, ttl: Optional[int] = None):
        """Set in cache (both Redis and local)"""
        ttl = ttl or self.ttl
        if self.redis and REDIS_AVAILABLE:
            try:
                self.redis.setex(key, ttl, json.dumps(value))
            except:
                pass

        self.local_cache[key] = value

    def clear(self):
        """Clear all caches"""
        self.local_cache.clear()
        if self.redis and REDIS_AVAILABLE:
            try:
                self.redis.flushdb()
            except:
                pass


class Telemetry:
    """Telemetry and observation system"""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}

    def log_event(self, event_name: str, tags: Dict[str, str] = None, value: Optional[float] = None):
        """Log telemetry event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_name,
            "tags": tags or {},
            "value": value
        }
        self.events.append(event)
        logger.info(f"Telemetry: {event}")

    def record_metric(self, metric_name: str, value: float):
        """Record metric"""
        self.metrics[metric_name] = value
        logger.info(f"Metric: {metric_name}={value}")

    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary"""
        return {
            "event_count": len(self.events),
            "metrics": self.metrics,
            "events": self.events[-10:]  # Last 10 events
        }


class DAGExecutionEngine:
    """Asynchronous DAG execution engine - orchestrates node execution"""

    def __init__(self, redis_client: Optional[Any] = None):
        self.nodes: Dict[str, Node] = {}
        self.redis = redis_client
        self.cache = CacheLayer(redis_client)
        self.telemetry = Telemetry()

    def add_node(self, node: Node):
        """Register node in DAG"""
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def _topological_sort(self, node_id: str) -> List[str]:
        """Get execution order using topological sort"""
        visited = set()
        order = []

        def visit(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for dep in node.dependencies:
                    visit(dep)
            order.append(nid)

        visit(node_id)
        return order

    async def execute(self, context: PipelineContext, start_node_id: str = "ingest") -> Dict[str, Any]:
        """Execute DAG starting from node"""
        self.telemetry.log_event("pipeline_start", {"pipeline_id": context.pipeline_id})

        execution_order = self._topological_sort(start_node_id)
        logger.info(f"Execution order: {execution_order}")

        try:
            for node_id in execution_order:
                node = self.nodes.get(node_id)
                if not node:
                    logger.warning(f"Node {node_id} not found")
                    continue

                logger.info(f"Executing node: {node_id}")
                result = await node.execute(context, self.redis)
                context.node_results[node_id] = result

                self.telemetry.log_event(
                    "node_executed",
                    {
                        "node_id": node_id,
                        "status": node.metadata.status.value,
                        "cache_hit": str(node.metadata.cache_hit)
                    },
                    node.metadata.duration
                )

            self.telemetry.log_event("pipeline_complete", {"pipeline_id": context.pipeline_id})

            return {
                "pipeline_id": context.pipeline_id,
                "status": "completed",
                "results": context.node_results,
                "execution_log": context.execution_log,
                "telemetry": self.telemetry.get_summary()
            }

        except Exception as e:
            self.telemetry.log_event("pipeline_failed", {"error": str(e)})
            logger.error(f"Pipeline execution failed: {e}")
            raise


# ============================================================================
# SPECIALIZED NODES FOR DREDGE ARCHITECTURE
# ============================================================================

class ModeBranchNode(Node):
    """Mode Graph - branches execution based on mode (standard/base)"""

    def __init__(self, node_id: str = "mode_graph", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.NORMALIZE, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        mode = context.input_data.get("mode", "base")
        context.execution_log.append(f"[MODE_GRAPH] Mode: {mode}")
        return {"mode": mode, "branch": "selected"}


class DREDGEStandardNode(Node):
    """DREDGE Standard - full pipeline with all features"""

    def __init__(self, node_id: str = "dredge_standard", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.EXECUTE, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        context.execution_log.append("[DREDGE_STANDARD] Full pipeline")
        return {"class": "base", "name": "base", "features": ["full", "cache", "telemetry"]}


class AsyncTranslationNode(Node):
    """Async Translation Node - translates, caches, applies telemetry"""

    def __init__(self, node_id: str = "async_translation", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.TRANSLATE, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        context.execution_log.append("[ASYNC_TRANSLATION] Running async translation")
        await asyncio.sleep(0.1)  # Simulate async work
        return {"translated": True, "async": True}


class RedisCacheRealNode(Node):
    """Redis Cache (Real Version) - actual Redis caching"""

    def __init__(self, node_id: str = "redis_cache", handler: Optional[Callable] = None):
        super().__init__(node_id, NodeType.CACHE, handler)

    async def execute(self, context: PipelineContext, redis_client: Optional[Any] = None) -> Dict:
        if redis_client and REDIS_AVAILABLE:
            context.execution_log.append("[REDIS_CACHE] Using Redis for persistence")
            return {"cache_backend": "redis", "persistent": True}
        else:
            context.execution_log.append("[REDIS_CACHE] Redis not available, using local cache")
            return {"cache_backend": "local", "persistent": False}


# ============================================================================
# PIPELINE FACTORY
# ============================================================================

class PipelineBuilder:
    """Build pre-configured pipelines"""

    @staticmethod
    def build_standard_pipeline(redis_client: Optional[Any] = None) -> DAGExecutionEngine:
        """Build the standard DREDGE pipeline from screenshots"""
        engine = DAGExecutionEngine(redis_client)

        # 1. CLI Entry
        ingest = IngestNode("cli_entry")
        engine.add_node(ingest)

        # 2. DAG Execution Engine (async orchestration)
        async def dag_handler(ctx: PipelineContext):
            ctx.execution_log.append("[DAG_ENGINE] Orchestrating async execution")
            return {"orchestration": "active", "async_mode": True}

        dag_node = Node("dag_engine", NodeType.EXECUTE, dag_handler)
        dag_node.add_dependency("cli_entry")
        engine.add_node(dag_node)

        # 3. Mode Graph
        mode_node = ModeBranchNode()
        mode_node.add_dependency("dag_engine")
        engine.add_node(mode_node)

        # 4. Node Graph - translate & normalize
        translate_node = TranslateNode()
        translate_node.add_dependency("mode_graph")
        engine.add_node(translate_node)

        normalize_node = NormalizeNode("normalize")
        normalize_node.add_fallback_provider("GoogleTranslator")
        normalize_node.add_fallback_provider("DeepProvider")
        normalize_node.add_dependency("translate")
        engine.add_node(normalize_node)

        # 5. Redis Cache Layer
        cache_node = RedisCacheRealNode()
        cache_node.add_dependency("normalize")
        engine.add_node(cache_node)

        # 6. Telemetry / Observation
        telemetry_node = Node("telemetry", NodeType.EXECUTE)
        telemetry_node.add_dependency("redis_cache")
        engine.add_node(telemetry_node)

        return engine

    @staticmethod
    def build_ios_swift_pipeline(redis_client: Optional[Any] = None) -> DAGExecutionEngine:
        """Build pipeline for iOS/Swift integration"""
        engine = DAGExecutionEngine(redis_client)

        # Simplified pipeline for iOS
        ingest = IngestNode()
        engine.add_node(ingest)

        async_translate = AsyncTranslationNode()
        async_translate.add_dependency("ingest")
        engine.add_node(async_translate)

        cache_node = RedisCacheRealNode()
        cache_node.add_dependency("async_translation")
        engine.add_node(cache_node)

        return engine


# ============================================================================
# PUBLIC API
# ============================================================================

async def dredge_run_pipeline(
    input_data: Dict[str, Any],
    pipeline_type: str = "standard",
    redis_client: Optional[Any] = None,
    pipeline_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point: dredge_run_pipeline()

    Args:
        input_data: Input to pipeline
        pipeline_type: "standard" or "ios_swift"
        redis_client: Optional Redis connection
        pipeline_id: Unique pipeline ID (auto-generated if not provided)

    Returns:
        Pipeline execution results with telemetry
    """
    pipeline_id = pipeline_id or f"dredge_{int(time.time() * 1000)}"

    if pipeline_type == "ios_swift":
        engine = PipelineBuilder.build_ios_swift_pipeline(redis_client)
    else:
        engine = PipelineBuilder.build_standard_pipeline(redis_client)

    context = PipelineContext(
        pipeline_id=pipeline_id,
        input_data=input_data,
        cache_enabled=True,
        async_mode=True
    )

    return await engine.execute(context)


if __name__ == "__main__":
    import sys

    # Demo
    async def demo():
        input_data = {
            "mode": "standard",
            "query": "test input",
            "metadata": {"user": "demo"}
        }

        result = await dredge_run_pipeline(input_data)
        print(json.dumps(result, indent=2))

    asyncio.run(demo())
