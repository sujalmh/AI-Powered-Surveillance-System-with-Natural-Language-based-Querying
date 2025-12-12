import traceback

print("=== FastAPI route introspection ===")
try:
    from backend.app.main import app
    paths = sorted({r.path for r in app.routes})
    print("Routes:", paths)
except Exception as e:
    print("Failed to import backend.app.main.app:", repr(e))
    traceback.print_exc()

print("\n=== Try importing chat router directly ===")
try:
    import importlib
    chat_mod = importlib.import_module("backend.app.routers.chat")
    print("chat router import: OK")
except Exception as e:
    print("chat router import: FAILED", repr(e))
    traceback.print_exc()

print("\n=== Try importing semantic router directly ===")
try:
    import importlib
    sem_mod = importlib.import_module("backend.app.routers.semantic")
    print("semantic router import: OK")
except Exception as e:
    print("semantic router import: FAILED", repr(e))
    traceback.print_exc()

print("\n=== Done ===")
