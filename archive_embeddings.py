import os
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_single_file(f, embed_dir):
    h = f.split(".")[0]
    path = os.path.join(embed_dir, f)
    try:
        # Load and ensure it's cloned to free zip storage references
        tensor = torch.load(path, weights_only=True).clone()
        return h, tensor
    except Exception as e:
        print(f"Error loading {f}: {e}")
        return h, None

def main():
    embed_dir = "./Dataset/Embeddings"
    archive_path = "./Dataset/all_embeddings.pt"
    
    if not os.path.exists(embed_dir):
        print("Embeddings directory not found!")
        return
        
    files = [f for f in os.listdir(embed_dir) if f.endswith(".pt")]
    print(f"Found {len(files)} individual embedding files in {embed_dir}")
    
    archive = {}
    print("Archiving embeddings into memory using ThreadPoolExecutor...")
    
    # We use 32 threads to speed up the disk I/O significantly on Windows
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(load_single_file, f, embed_dir): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Loading"):
            h, tensor = future.result()
            if tensor is not None:
                archive[h] = tensor
                
    print(f"Loaded {len(archive)} embeddings. Saving archive to {archive_path}...")
    torch.save(archive, archive_path)
    print("Archive saved successfully!")

if __name__ == "__main__":
    main()
