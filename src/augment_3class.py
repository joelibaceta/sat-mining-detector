"""
Augment dataset with 3 classes by rotating underrepresented categories
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def rotate_image(image, k):
    """Rotate image by k*90 degrees"""
    return np.rot90(image, k=k, axes=(0, 1))

def augment_dataset_3class(input_dir, output_dir):
    """
    Augment dataset maintaining 3 classes and balancing them
    
    Current distribution:
    - mining_emergence: 366 samples
    - natural_change: 87 samples  
    - human_activity: 70 samples
    
    Strategy:
    - mining: 1x (original only) = 366
    - natural: 4x (0°, 90°, 180°, 270°) = 87 * 4 = 348
    - human: 5x (original + 4 more augmentations) = 70 * 5 = 350
    
    Final: ~366, 348, 350 (balanced)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Load original dataset
    train_df = pd.read_csv(input_dir / 'dataset_train.csv')
    val_df = pd.read_csv(input_dir / 'dataset_val.csv')
    test_df = pd.read_csv(input_dir / 'dataset_test.csv')
    
    # Count by category
    print("\nOriginal distribution:")
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n{split_name.upper()}:")
        counts = df['category'].value_counts()
        for cat, count in counts.items():
            print(f"  {cat}: {count}")
    
    # Define augmentation strategy
    augmentation_map = {
        'mining_emergence': [0],  # Only original (no rotation)
        'natural_change': [0, 1, 2, 3],  # 4x: 0°, 90°, 180°, 270°
        'human_activity': [0, 1, 2, 3, 1]  # 5x: original + 4 rotations (repeat 90° to get 5)
    }
    
    # Label mapping
    label_map = {
        'mining_emergence': 0,
        'human_activity': 1,
        'natural_change': 2
    }
    
    def process_split(df, split_name, output_dir):
        """Process one split with augmentation"""
        augmented_rows = []
        t0_output = output_dir / split_name / 't0'
        t1_output = output_dir / split_name / 't1'
        t0_output.mkdir(parents=True, exist_ok=True)
        t1_output.mkdir(parents=True, exist_ok=True)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            category = row['category']
            sample_id = row['sample_id']
            
            # Get rotation strategy for this category
            rotations = augmentation_map.get(category, [0])
            
            # Load original images
            t0_path = input_dir / row['t0_path']
            t1_path = input_dir / row['t1_path']
            
            if not t0_path.exists() or not t1_path.exists():
                print(f"Warning: Missing files for {sample_id}")
                continue
                
            t0_img = np.load(t0_path)
            t1_img = np.load(t1_path)
            
            # Apply rotations
            for rot_idx, k in enumerate(rotations):
                # Rotate both images with same angle
                t0_rotated = rotate_image(t0_img, k)
                t1_rotated = rotate_image(t1_img, k)
                
                # Save rotated images
                rot_suffix = f"rot{k*90}"
                t0_save_path = t0_output / f"{sample_id}_t0_{rot_suffix}.npy"
                t1_save_path = t1_output / f"{sample_id}_t1_{rot_suffix}.npy"
                
                np.save(t0_save_path, t0_rotated)
                np.save(t1_save_path, t1_rotated)
                
                # Add to augmented dataset
                augmented_rows.append({
                    't0_path': f"{split_name}/t0/{t0_save_path.name}",
                    't1_path': f"{split_name}/t1/{t1_save_path.name}",
                    'label': label_map[category],
                    'category': category,
                    'original_idx': idx,
                    'rotation': rot_suffix,
                    'sample_id': sample_id
                })
        
        return pd.DataFrame(augmented_rows)
    
    # Process each split
    print("\nAugmenting datasets...")
    train_aug = process_split(train_df, 'train', output_dir)
    val_aug = process_split(val_df, 'val', output_dir)
    test_aug = process_split(test_df, 'test', output_dir)
    
    # Save augmented CSVs
    train_aug.to_csv(output_dir / 'dataset_train.csv', index=False)
    val_aug.to_csv(output_dir / 'dataset_val.csv', index=False)
    test_aug.to_csv(output_dir / 'dataset_test.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    
    for split_name, df in [('train', train_aug), ('val', val_aug), ('test', test_aug)]:
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {len(df)}")
        counts = df['category'].value_counts()
        for cat, count in counts.items():
            label = label_map[cat]
            pct = 100 * count / len(df)
            print(f"  {cat} (label={label}): {count} ({pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Augment dataset with 3 classes')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for augmented dataset')
    
    args = parser.parse_args()
    
    augment_dataset_3class(args.input_dir, args.output_dir)
    print(f"\nAugmented dataset saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
