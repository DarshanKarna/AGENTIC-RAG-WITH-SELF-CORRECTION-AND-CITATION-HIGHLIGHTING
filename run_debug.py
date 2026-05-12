import traceback
try:
    import baseline_rag
    baseline_rag.main()
except Exception as e:
    traceback.print_exc()
    print(f"ERROR: {e}")
