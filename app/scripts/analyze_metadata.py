"""
Analyze Amazon Fashion metadata to understand field usage and prepare for embeddings.
"""
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any
from pathlib import Path


def analyze_product_metadata(data_path: str) -> Dict[str, Any]:
    """
    Analyze product metadata to understand field usage patterns.
    """
    with open(data_path, 'r') as f:
        products = json.load(f)
    
    total_products = len(products)
    
    # Initialize counters
    field_counts = defaultdict(int)
    category_counts = Counter()
    store_counts = Counter()
    detail_key_counts = Counter()
    features_length = []
    description_length = []
    images_count = []
    rating_distribution = defaultdict(int)
    price_distribution = {'null': 0, 'has_price': 0}
    
    # Fields that contain text content for embeddings
    text_fields_populated = {
        'title': 0,
        'features': 0,
        'description': 0,
        'store': 0,
        'categories': 0,
        'details': 0
    }
    
    for product in products:
        # Count field presence
        for field in product.keys():
            if product.get(field) is not None:
                field_counts[field] += 1
        
        # Analyze main_category
        if product.get('main_category'):
            category_counts[product['main_category']] += 1
        
        # Analyze store
        if product.get('store'):
            store_counts[product['store']] += 1
            text_fields_populated['store'] += 1
        
        # Analyze title
        if product.get('title'):
            text_fields_populated['title'] += 1
        
        # Analyze features
        if product.get('features'):
            features_length.append(len(product['features']))
            if len(product['features']) > 0:
                text_fields_populated['features'] += 1
        
        # Analyze description
        if product.get('description'):
            description_length.append(len(product['description']))
            if len(product['description']) > 0:
                text_fields_populated['description'] += 1
        
        # Analyze categories
        if product.get('categories') and len(product['categories']) > 0:
            text_fields_populated['categories'] += 1
        
        # Analyze details
        if product.get('details'):
            text_fields_populated['details'] += 1
            for key in product['details'].keys():
                detail_key_counts[key] += 1
        
        # Analyze images
        if product.get('images'):
            images_count.append(len(product['images']))
        
        # Analyze ratings
        if product.get('average_rating') is not None:
            rating = round(product['average_rating'])
            rating_distribution[rating] += 1
        
        # Analyze price
        if product.get('price') is not None:
            price_distribution['has_price'] += 1
        else:
            price_distribution['null'] += 1
    
    # Calculate statistics
    stats = {
        'total_products': total_products,
        'field_presence': {field: count for field, count in field_counts.items()},
        'field_presence_percentage': {
            field: f"{(count/total_products)*100:.1f}%" 
            for field, count in field_counts.items()
        },
        'text_fields_populated': text_fields_populated,
        'text_fields_percentage': {
            field: f"{(count/total_products)*100:.1f}%" 
            for field, count in text_fields_populated.items()
        },
        'category_distribution': dict(category_counts.most_common(10)),
        'store_distribution': dict(store_counts.most_common(10)),
        'detail_keys_frequency': dict(detail_key_counts.most_common(15)),
        'features_stats': {
            'products_with_features': len(features_length),
            'avg_features_count': sum(features_length) / len(features_length) if features_length else 0,
            'max_features': max(features_length) if features_length else 0,
            'min_features': min(features_length) if features_length else 0
        },
        'description_stats': {
            'products_with_description': len(description_length),
            'avg_description_count': sum(description_length) / len(description_length) if description_length else 0,
            'max_descriptions': max(description_length) if description_length else 0
        },
        'images_stats': {
            'products_with_images': len(images_count),
            'avg_images_count': sum(images_count) / len(images_count) if images_count else 0,
            'max_images': max(images_count) if images_count else 0
        },
        'rating_distribution': dict(rating_distribution),
        'price_distribution': price_distribution
    }
    
    return stats


def suggest_embedding_combinations(stats: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Suggest different text combinations for embeddings based on field availability.
    """
    combinations = []
    
    # Check field population rates
    text_fields = stats['text_fields_populated']
    
    # Combination 1: Title only (baseline - should work for all products)
    combinations.append({
        'name': 'title_only',
        'description': 'Product title only',
        'fields': ['title'],
        'coverage': stats['text_fields_percentage']['title']
    })
    
    # Combination 2: Title + Features (if features are well populated)
    if text_fields['features'] > text_fields['title'] * 0.5:
        combinations.append({
            'name': 'title_features',
            'description': 'Title combined with product features',
            'fields': ['title', 'features'],
            'coverage': f"~{min(float(stats['text_fields_percentage']['title'].rstrip('%')), float(stats['text_fields_percentage']['features'].rstrip('%'))):.1f}%"
        })
    
    # Combination 3: Title + Description (if descriptions exist)
    if text_fields['description'] > 50:
        combinations.append({
            'name': 'title_description',
            'description': 'Title with product description',
            'fields': ['title', 'description'],
            'coverage': f"~{min(float(stats['text_fields_percentage']['title'].rstrip('%')), float(stats['text_fields_percentage']['description'].rstrip('%'))):.1f}%"
        })
    
    # Combination 4: Title + Categories + Store
    combinations.append({
        'name': 'title_category_store',
        'description': 'Title with category and store information',
        'fields': ['title', 'main_category', 'store'],
        'coverage': f"~{min(float(stats['text_fields_percentage']['title'].rstrip('%')), float(stats['text_fields_percentage']['store'].rstrip('%'))):.1f}%"
    })
    
    # Combination 5: Kitchen sink (all available text)
    combinations.append({
        'name': 'all_text',
        'description': 'All available text fields combined',
        'fields': ['title', 'features', 'description', 'main_category', 'store', 'categories'],
        'coverage': 'Varies by product'
    })
    
    # Combination 6: Title + Selected Details
    combinations.append({
        'name': 'title_details',
        'description': 'Title with selected product details (brand, material, etc.)',
        'fields': ['title', 'details_selected'],
        'coverage': f"~{float(stats['text_fields_percentage']['details'].rstrip('%')):.1f}%"
    })
    
    return combinations


def print_analysis_report(stats: Dict[str, Any], combinations: List[Dict[str, str]]):
    """
    Print a formatted analysis report.
    """
    print("=" * 80)
    print("AMAZON FASHION METADATA ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal Products: {stats['total_products']}")
    
    print("\n" + "-" * 40)
    print("FIELD PRESENCE ANALYSIS")
    print("-" * 40)
    for field, percentage in stats['field_presence_percentage'].items():
        print(f"{field:20} {percentage:>8} ({stats['field_presence'][field]} products)")
    
    print("\n" + "-" * 40)
    print("TEXT FIELDS POPULATION")
    print("-" * 40)
    for field, percentage in stats['text_fields_percentage'].items():
        print(f"{field:20} {percentage:>8} ({stats['text_fields_populated'][field]} products)")
    
    print("\n" + "-" * 40)
    print("CATEGORY DISTRIBUTION (Top 10)")
    print("-" * 40)
    for category, count in stats['category_distribution'].items():
        print(f"{category:30} {count:>5} products")
    
    print("\n" + "-" * 40)
    print("STORE DISTRIBUTION (Top 10)")
    print("-" * 40)
    for store, count in stats['store_distribution'].items():
        print(f"{store[:40]:40} {count:>5} products")
    
    print("\n" + "-" * 40)
    print("PRODUCT DETAILS KEYS (Top 15)")
    print("-" * 40)
    for key, count in stats['detail_keys_frequency'].items():
        print(f"{key:40} {count:>5} occurrences")
    
    print("\n" + "-" * 40)
    print("FEATURES STATISTICS")
    print("-" * 40)
    for key, value in stats['features_stats'].items():
        if isinstance(value, float):
            print(f"{key:30} {value:.2f}")
        else:
            print(f"{key:30} {value}")
    
    print("\n" + "-" * 40)
    print("DESCRIPTION STATISTICS")
    print("-" * 40)
    for key, value in stats['description_stats'].items():
        if isinstance(value, float):
            print(f"{key:30} {value:.2f}")
        else:
            print(f"{key:30} {value}")
    
    print("\n" + "-" * 40)
    print("PRICE DISTRIBUTION")
    print("-" * 40)
    for key, value in stats['price_distribution'].items():
        print(f"{key:30} {value} products")
    
    print("\n" + "=" * 80)
    print("SUGGESTED EMBEDDING COMBINATIONS")
    print("=" * 80)
    for i, combo in enumerate(combinations, 1):
        print(f"\n{i}. {combo['name'].upper()}")
        print(f"   Description: {combo['description']}")
        print(f"   Fields: {', '.join(combo['fields'])}")
        print(f"   Coverage: {combo['coverage']}")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "amazon_fashion_sample.json"
    
    # Analyze metadata
    stats = analyze_product_metadata(str(data_path))
    
    # Get embedding combination suggestions
    combinations = suggest_embedding_combinations(stats)
    
    # Print report
    print_analysis_report(stats, combinations)
    
    # Save stats to JSON for later use
    output_path = Path(__file__).parent.parent.parent / "data" / "metadata_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            'stats': stats,
            'embedding_combinations': combinations
        }, f, indent=2)
    
    print(f"\n\nAnalysis saved to: {output_path}")