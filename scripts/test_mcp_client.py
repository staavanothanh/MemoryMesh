import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def main():
    server_params = StdioServerParameters(
        command="memorymesh",  # Lệnh chạy server
        args=[],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("✅ Tools:", [t.name for t in tools.tools])

            # Gọi thử ping
            result = await session.call_tool("ping", {})
            print("✅ Ping:", result.content[0].text)

            # Gọi remember
            result = await session.call_tool("remember", {"content": "Hà Nội là thủ đô"})
            print("✅ Remember:", result.content[0].text)

            # Gọi recall
            result = await session.call_tool("recall", {"query": "thủ đô"})
            print("✅ Recall:", result.content[0].text)

asyncio.run(main())