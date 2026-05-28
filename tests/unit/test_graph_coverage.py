"""Test uncovered paths in graph_store.py:
delete_entity, merge_entities, search_entities, format_xml_triplet with various inputs,
list_entities with limit/offset.

Uses the same pattern as test_graph_store.py: raw aiosqlite connection + asyncio.Lock.
"""

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


@pytest.mark.asyncio
async def test_delete_entity(db):
    """delete_entity removes entity and its relations."""
    _, graph = db
    eid = await graph.create_entity(
        name="to-delete", user_id="test_user", entity_type="concept"
    )
    result = await graph.get_entity_by_id(eid)
    assert result is not None

    success = await graph.delete_entity(eid, "test_user")
    assert success is True

    result = await graph.get_entity_by_id(eid)
    assert result is None


@pytest.mark.asyncio
async def test_delete_entity_nonexistent(db):
    """delete_entity returns False for non-existent entity."""
    _, graph = db
    success = await graph.delete_entity("nonexistent-id", "test_user")
    assert success is False


@pytest.mark.asyncio
async def test_delete_entity_removes_relations(db):
    """Deleting an entity also removes related relations."""
    _, graph = db
    src_id = await graph.create_entity(
        name="source", user_id="test_user", entity_type="concept"
    )
    tgt_id = await graph.create_entity(
        name="target", user_id="test_user", entity_type="concept"
    )
    await graph.create_relation(
        source_id=src_id, target_id=tgt_id,
        relation_type="depends_on", user_id="test_user",
    )

    # Delete source — relations should be removed
    await graph.delete_entity(src_id, "test_user")

    # Relations for target should be empty
    rels = await graph.get_relations_for_entity(tgt_id, "test_user")
    assert len(rels) == 0


@pytest.mark.asyncio
async def test_search_entities_by_name(db):
    """search_entities_fts finds entities by name pattern."""
    _, graph = db
    await graph.create_entity(name="Alpha Project", user_id="test_user")
    await graph.create_entity(name="Beta Module", user_id="test_user")
    await graph.create_entity(name="Gamma Config", user_id="test_user")

    results = await graph.search_entities_fts("alpha", "test_user")
    assert len(results) >= 1
    assert any("Alpha" in r["name"] for r in results)

    results = await graph.search_entities_fts("Module", "test_user")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_search_entities_with_limit(db):
    """search_entities_fts respects limit parameter."""
    _, graph = db
    for i in range(5):
        await graph.create_entity(
            name=f"Searchable Entity {i}", user_id="test_user"
        )

    results = await graph.search_entities_fts("Searchable", "test_user", limit=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_merge_entities(db):
    """merge_entities migrates relations from source to target."""
    _, graph = db
    src_id = await graph.create_entity(
        name="source-entity", user_id="test_user", properties={"key": "src"}
    )
    tgt_id = await graph.create_entity(
        name="target-entity", user_id="test_user", properties={"key": "tgt"}
    )
    other_id = await graph.create_entity(
        name="other-entity", user_id="test_user"
    )

    # Create relation from source to other
    await graph.create_relation(
        source_id=src_id, target_id=other_id,
        relation_type="depends_on", user_id="test_user",
    )

    # Merge source -> target
    result = await graph.merge_entities("source-entity", "target-entity", "test_user")
    assert result["success"] is True
    assert result["relations_migrated"] >= 1

    # Source entity is gone
    source = await graph.get_entity_by_name("source-entity", "test_user")
    assert source is None

    # Target now has the relations
    rels = await graph.get_relations_for_entity(tgt_id, "test_user")
    assert len(rels) >= 1


@pytest.mark.asyncio
async def test_merge_entities_nonexistent_source(db):
    """merge_entities returns error when source doesn't exist."""
    _, graph = db
    result = await graph.merge_entities(
        "nonexistent-source", "target-entity", "test_user"
    )
    assert result["success"] is False
    assert "not found" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_format_xml_triplet(db):
    """format_xml_triplet produces valid XML."""
    _, graph = db
    src_id = await graph.create_entity(name="XML Source", user_id="test_user")
    tgt_id = await graph.create_entity(name="XML Target", user_id="test_user")
    await graph.create_relation(
        source_id=src_id, target_id=tgt_id,
        relation_type="depends_on", user_id="test_user",
        weight=0.9,
    )
    rels = await graph.get_relations_for_entity(src_id, "test_user")

    xml = graph.format_xml_triplet(rels)
    assert "<KnowledgeGraph>" in xml
    assert "XML Source" in xml
    assert "XML Target" in xml
    assert "depends_on" in xml


@pytest.mark.asyncio
async def test_format_xml_triplet_empty(db):
    """format_xml_triplet returns empty string for empty list."""
    _, graph = db
    xml = graph.format_xml_triplet([])
    assert xml == ""


@pytest.mark.asyncio
async def test_format_xml_from_neighbors(db):
    """format_xml_from_neighbors produces proper XML."""
    _, graph = db
    src_id = await graph.create_entity(name="Neighbor Source", user_id="test_user")
    tgt_id = await graph.create_entity(name="Neighbor Target", user_id="test_user")
    await graph.create_relation(
        source_id=src_id, target_id=tgt_id,
        relation_type="connected_to", user_id="test_user",
    )

    result = await graph.query_graph("Neighbor Source", "test_user")
    xml = graph.format_xml_from_neighbors(
        result.get("entity"), result.get("neighbors", [])
    )
    assert "<KnowledgeGraph" in xml
    assert "Neighbor Source" in xml
    assert "connected_to" in xml


@pytest.mark.asyncio
async def test_list_entities_with_limit(db):
    """list_entities respects limit."""
    _, graph = db
    for i in range(5):
        await graph.create_entity(name=f"List Entity {i}", user_id="test_user")

    all_entities = await graph.list_entities("test_user", limit=3)
    assert len(all_entities) <= 3


@pytest.mark.asyncio
async def test_get_entity_by_name_not_found(db):
    """get_entity_by_name returns None for non-existent entity."""
    _, graph = db
    result = await graph.get_entity_by_name("non-existent-name", "test_user")
    assert result is None


@pytest.mark.asyncio
async def test_get_entity_by_id_not_found(db):
    """get_entity_by_id returns None for non-existent ID."""
    _, graph = db
    result = await graph.get_entity_by_id("non-existent-id")
    assert result is None
