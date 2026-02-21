
try:
    from raptor import RetrievalAugmentation
    print("Raptor imported")
    ra = RetrievalAugmentation()
    print("Raptor methods:", dir(ra))
except ImportError:
    print("Raptor import failed")
except Exception as e:
    print(f"Raptor error: {e}")

try:
    import graphrag.query.structured_search.local_search.search as ls
    print("GraphRAG LocalSearch imported from search")
except ImportError:
    try:
        from graphrag.query.structured_search.local_search.search import LocalSearch
        print("GraphRAG LocalSearch imported directly")
    except ImportError:
        print("GraphRAG LocalSearch import failed")

try:
    import graphrag
    print(f"GraphRAG version: {graphrag.__version__}")
except:
    pass
