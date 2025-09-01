import json
import urllib.request
import os

def download_and_process_fashion_data():
    """
    Download Amazon Fashion metadata from the provided Hugging Face URL
    """
    url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Amazon_Fashion.jsonl"
    local_file = "meta_Amazon_Fashion.jsonl"
    
    # Download if not exists
    if not os.path.exists(local_file):
        print(f"Downloading Amazon Fashion metadata...")
        print(f"URL: {url}")
        print("This may take a while...")
        
        urllib.request.urlretrieve(url, local_file)
        print(f"Downloaded to {local_file}")
    else:
        print(f"Using existing file: {local_file}")
    
    # Read and process the data
    print("\nReading first 300 items...")
    
    items = []
    with open(local_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 300:
                break
            
            item = json.loads(line.strip())
            items.append(item)
            
            # Show progress
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1} items...")
            
            # Show structure of first 5 items
            if i < 5:
                print(f"\nItem {i + 1}:")
                print("-" * 40)
                for key, value in item.items():
                    if isinstance(value, (list, dict)):
                        print(f"{key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                        if isinstance(value, list) and len(value) > 0:
                            sample = str(value[0])[:80] + "..." if len(str(value[0])) > 80 else str(value[0])
                            print(f"  Sample: {sample}")
                    else:
                        display_val = str(value)[:80] + "..." if value and len(str(value)) > 80 else str(value)
                        print(f"{key}: {display_val}")
    
    print(f"\nTotal items loaded: {len(items)}")
    
    # Save to JSON
    output_file = "amazon_fashion_sample.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")
    
    # Also save as JSONL
    output_file_jsonl = "amazon_fashion_sample.jsonl"
    with open(output_file_jsonl, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Also saved to {output_file_jsonl}")
    
    # Show summary
    if items:
        print("\n" + "=" * 80)
        print("Data structure summary:")
        print(f"Fields: {list(items[0].keys())}")
        print(f"Number of fields per item: {len(items[0])}")
    
    return items

if __name__ == "__main__":
    print("=" * 80)
    print("Amazon Fashion Metadata Downloader")
    print("=" * 80)
    
    items = download_and_process_fashion_data()
    
    if items:
        print("\n✓ Success! Loaded and saved 300 Amazon Fashion metadata items")
        print("\nFiles created:")
        print("  - amazon_fashion_sample.json (JSON format)")
        print("  - amazon_fashion_sample.jsonl (JSONL format)")
    else:
        print("\n✗ Failed to load data")