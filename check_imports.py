try:
    from transformers import Sam3Model
    print("Sam3Model found in transformers")
except ImportError as e:
    print(f"Error: {e}")

try:
    import sam3
    print("sam3 package found")
except ImportError as e:
    print(f"Error: {e}")
