from app.agent import agent

result = agent.query("Que es BigQuery y para que se usa?", thread_id="test-1")

print("RESPUESTA:")
print(result["answer"])
print()
print("FUENTES:")
for s in result["sources"]:
    print(f"  - {s['title']} | {s['url']}")
print()
print("RELACIONADOS:")
for r in result["related"]:
    print(f"  - {r['title']}")