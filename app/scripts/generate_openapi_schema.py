#!/usr/bin/env python3
"""
Generate OpenAPI schema for TypeScript generation.
Use with packages like openapi-typescript or swagger-codegen.
"""

import json
from app.app import app


def generate_openapi_schema():
    """Generate OpenAPI schema JSON file"""
    schema = app.openapi()
    
    # Write to file
    with open("openapi.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    print("✅ OpenAPI schema generated: openapi.json")
    print("\nTo generate TypeScript types, install and use:")
    print("  npm install -g openapi-typescript")
    print("  npx openapi-typescript openapi.json -o types.ts")
    print("\nOr use swagger-codegen:")
    print("  npm install -g @openapitools/openapi-generator-cli")
    print("  npx @openapitools/openapi-generator-cli generate -i openapi.json -g typescript-fetch -o ./generated")


if __name__ == "__main__":
    generate_openapi_schema()