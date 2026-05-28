import asyncio
import json
import pytest
import pytest_asyncio

import aiosqlite

from memorymesh.memory.graph_store import GraphStore


@pytest_asyncio.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    graph = GraphStore(conn, asyncio.Lock())
    await graph.create_schema()
    yield (conn, graph)
    await conn.close()


class TestGraphSchema:
    async def test_schema_creates_tables(self, db):
        conn, _ = db
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('entities', 'relations')"
        )
        rows = await cursor.fetchall()
        assert len(rows) == 2


class TestCreateEntity:
    async def test_create_entity_returns_id(self, db):
        _, graph = db
        entity_id = await graph.create_entity("TestConcept", "user1")
        assert entity_id is not None
        assert len(entity_id) > 0

    async def test_create_entity_with_type_and_properties(self, db):
        _, graph = db
        props = {"summary": "A test concept"}
        entity_id = await graph.create_entity("BugFix", "user1", entity_type="bug", properties=props)
        entity = await graph.get_entity_by_name("BugFix", "user1")
        assert entity is not None
        assert entity["type"] == "bug"
        assert entity["properties"] == props

    async def test_list_entities(self, db):
        _, graph = db
        await graph.create_entity("EntityA", "user1")
        await graph.create_entity("EntityB", "user1")
        entities = await graph.list_entities("user1")
        assert len(entities) == 2


class TestCreateRelation:
    async def test_create_relation_links_entities(self, db):
        _, graph = db
        src_id = await graph.create_entity("Source", "user1")
        tgt_id = await graph.create_entity("Target", "user1")
        rel_id = await graph.create_relation(src_id, tgt_id, "DEPENDS_ON", "user1")
        assert rel_id is not None

    async def test_get_relations_for_entity(self, db):
        _, graph = db
        src = await graph.create_entity("ModuleA", "user1")
        tgt = await graph.create_entity("ModuleB", "user1")
        await graph.create_relation(src, tgt, "USES", "user1")
        relations = await graph.get_relations_for_entity(src, "user1")
        assert len(relations) == 1
        assert relations[0]["relation_type"] == "USES"

    async def test_auto_creates_entities_on_relation(self, db):
        _, graph = db
        src_id = await graph.create_entity("PlanPhase1", "user1")
        tgt_id = await graph.create_entity("AuthBug", "user1")
        rel_id = await graph.create_relation(src_id, tgt_id, "SOLVES", "user1")
        assert rel_id is not None
        src = await graph.get_entity_by_id(src_id)
        tgt = await graph.get_entity_by_id(tgt_id)
        assert src["name"] == "PlanPhase1"
        assert tgt["name"] == "AuthBug"


class TestQueryGraph:
    async def test_query_1hop_neighbors(self, db):
        _, graph = db
        src = await graph.create_entity("Center", "user1")
        tgt = await graph.create_entity("Neighbor", "user1")
        await graph.create_relation(src, tgt, "CONNECTED_TO", "user1")
        result = await graph.query_graph("Center", "user1")
        assert result["entity"] is not None
        assert len(result["neighbors"]) == 1
        assert result["neighbors"][0]["entity"] == "Neighbor"
        assert result["neighbors"][0]["relation"] == "CONNECTED_TO"

    async def test_query_nonexistent_entity(self, db):
        _, graph = db
        result = await graph.query_graph("Ghost", "user1")
        assert result["entity"] is None
        assert "not found" in result.get("error", "")


class TestTraceEntity:
    async def test_multi_hop_traversal(self, db):
        _, graph = db
        a = await graph.create_entity("NodeA", "user1")
        b = await graph.create_entity("NodeB", "user1")
        c = await graph.create_entity("NodeC", "user1")
        await graph.create_relation(a, b, "LINKS_TO", "user1")
        await graph.create_relation(b, c, "LINKS_TO", "user1")
        result = await graph.trace_entity("NodeA", "user1", max_depth=3)
        assert result["entity"] is not None
        assert len(result["relations"]) == 2

    async def test_cyclic_graph_no_infinite_loop(self, db):
        _, graph = db
        a = await graph.create_entity("A", "user1")
        b = await graph.create_entity("B", "user1")
        await graph.create_relation(a, b, "LINKS_TO", "user1")
        await graph.create_relation(b, a, "LINKS_TO", "user1")
        result = await graph.trace_entity("A", "user1", max_depth=5)
        assert result["entity"] is not None
        assert len(result["relations"]) > 0


class TestFormat:
    async def test_format_xml_triplet_empty(self, db):
        _, graph = db
        assert graph.format_xml_triplet([]) == ""

    async def test_format_xml_triplet(self, db):
        _, graph = db
        src = await graph.create_entity("Plan", "user1")
        tgt = await graph.create_entity("Bug", "user1")
        await graph.create_relation(src, tgt, "SOLVES", "user1")
        relations = await graph.get_relations_for_entity(src, "user1")
        xml = graph.format_xml_triplet(relations)
        assert "<KnowledgeGraph>" in xml
        assert "SOLVES" in xml
        assert "Plan" in xml
        assert "Bug" in xml
        assert "</KnowledgeGraph>" in xml

    async def test_format_xml_from_neighbors(self, db):
        _, graph = db
        src = await graph.create_entity("Root", "user1")
        tgt = await graph.create_entity("Leaf", "user1")
        await graph.create_relation(src, tgt, "DEPENDS_ON", "user1")
        result = await graph.query_graph("Root", "user1")
        xml = graph.format_xml_from_neighbors(result["entity"], result["neighbors"])
        assert "<KnowledgeGraph" in xml
        assert "DEPENDS_ON" in xml
