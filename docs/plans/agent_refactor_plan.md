# Agent-Based Search Architecture Refactor Plan

## Overview
Refactor the search implementation from service-based to agent-based architecture where the fashion agent orchestrates search tools instead of API calling services directly.

## Current vs. Proposed Architecture

### Current (Service-Based)
```
POST /chat/search → Services → Database
├── IntentClassifier Service
├── RetrievalService  
├── RankingService
└── FacetsService
```

### Proposed (Agent-Based)
```
POST /chat/search → Fashion Agent → Tools → Database
                                  ├── intent_classification_tool
                                  ├── semantic_search_tool (existing)
                                  ├── ranking_tool
                                  └── facets_tool
```

## Implementation Steps

### Step 1: Create Agent Tools

#### 1.1 Intent Classification Tool (`app/agents/tools/intent_classifier.py`)
- Move LLM classification logic from `IntentClassifier` service
- Return structured `ParsedQuery` with complexity and filters
- Use settings for model configuration

#### 1.2 Ranking Tool (`app/agents/tools/ranking.py`) 
- Move Bayesian ranking logic from `RankingService`
- Use configurable parameters from `settings`
- Take list of candidates and return ranked results

#### 1.3 Facets Tool (`app/agents/tools/facets.py`)
- Extract facet generation from `FacetsService` 
- **Remove** follow-up prompt generation (not needed)
- Generate filter options from search results

### Step 2: Update Fashion Agent

#### 2.1 Agent Orchestration (`app/agents/fashion_agent.py`)
- Add search orchestration method
- Call tools in sequence: classify → retrieve → rank → facets
- Handle tool errors and fallbacks
- Return structured `ChatSearchResponse`

#### 2.2 Tool Registration
- Register all search tools with the agent
- Ensure proper tool calling with structured outputs

### Step 3: Update API Layer

#### 3.1 Search Endpoint (`app/api/search.py`)
- Replace service calls with fashion agent invocation
- Pass `SearchRequest` to agent
- Handle agent response and error cases
- Keep existing debug tracing and performance monitoring

#### 3.2 Maintain Response Format
- Keep existing `ChatSearchResponse` structure
- Ensure backward compatibility with frontend

### Step 4: Configuration Updates

#### 4.1 Tool Settings
- Tools should use settings for default parameters:
  - `settings.ranking_lambda_blend`
  - `settings.ranking_bayesian_mu` 
  - `settings.ranking_bayesian_w`
  - `settings.search_topk_default`
  - `settings.startup_embedding_strategy`

### Step 5: Clean Up Services

#### 5.1 Deprecate Service Classes
- Mark services as deprecated but keep for now
- Remove service imports from API layer
- Document migration path in docstrings

## Files to Create/Modify

### New Files
- `app/agents/tools/intent_classifier.py`
- `app/agents/tools/ranking.py` 
- `app/agents/tools/facets.py`

### Modified Files
- `app/agents/fashion_agent.py` - Add search orchestration
- `app/api/search.py` - Replace services with agent calls
- `app/agents/tools/__init__.py` - Export new tools

### Existing Files (Keep)
- `app/agents/tools/search.py` - Already implements semantic search
- `app/types/search_api.py` - Keep all type definitions
- `app/settings.py` - Already updated with ranking config

## Testing Strategy

### Unit Tests
- Test each tool independently
- Mock external dependencies (OpenAI, database)
- Verify tool outputs match expected types

### Integration Tests  
- Test full agent orchestration
- Verify API response format unchanged
- Test error handling and fallbacks

### Performance Tests
- Compare agent vs service response times
- Ensure no performance regression
- Test with various query complexities

## Migration Benefits

1. **Modularity**: Tools are reusable across different agents
2. **Testability**: Each tool can be tested in isolation  
3. **Flexibility**: Agent can adapt tool usage based on query complexity
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new search tools

## Risk Mitigation

- Keep services temporarily for rollback capability
- Implement feature flags for service vs agent switching
- Comprehensive testing before removing services
- Monitor performance and error rates post-migration

## Success Criteria

- [ ] All search functionality works through agent tools
- [ ] API response format unchanged
- [ ] Performance within 10% of current implementation  
- [ ] Error handling maintains current behavior
- [ ] Debug information still available
- [ ] Configuration parameters respected
