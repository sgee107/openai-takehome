import { NextRequest, NextResponse } from 'next/server';
import { searchProducts } from '../../../lib/mockData';
import { ChatResponse } from '../../../lib/types';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query = '', limit = 20 } = body;
    
    // Add artificial delay to show skeleton loading
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Search products based on query
    const results = searchProducts(query, limit);
    
    const response: ChatResponse = {
      results,
      query,
      strategy: 'mock_text_matching',
      total: results.length
    };
    
    return NextResponse.json(response);
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Failed to process search request' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ message: 'Fashion search API - use POST to search' });
}