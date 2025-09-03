"""
Strategy registration and management for text generation strategies.
"""
from typing import Dict, List, Type
from app.process.strategies.text_strategies import (
    BaseTextStrategy,
    TitleOnlyStrategy,
    TitleFeaturesStrategy,
    TitleCategoryStoreStrategy,
    TitleDetailsStrategy,
    ComprehensiveStrategy,
    KeyValueBasicStrategy,
    KeyValueDetailedStrategy,
    KeyValueWithImagesStrategy,
    KeyValueComprehensiveStrategy
)


class StrategyRegistry:
    """Registry for managing text generation strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, BaseTextStrategy] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register all default strategies."""
        # Original TextStrategy methods (converted to classes)
        self.register('title_only', TitleOnlyStrategy())
        self.register('title_features', TitleFeaturesStrategy())
        self.register('title_category_store', TitleCategoryStoreStrategy())
        self.register('title_details', TitleDetailsStrategy())
        self.register('comprehensive', ComprehensiveStrategy())
        
        # Key-value strategies
        self.register('key_value_basic', KeyValueBasicStrategy())
        self.register('key_value_detailed', KeyValueDetailedStrategy())
        self.register('key_value_with_images', KeyValueWithImagesStrategy())
        self.register('key_value_comprehensive', KeyValueComprehensiveStrategy())
    
    def register(self, name: str, strategy: BaseTextStrategy):
        """Register a strategy with a given name.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        self._strategies[name] = strategy
    
    def get_strategy(self, name: str) -> BaseTextStrategy:
        """Get a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance
            
        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            available = ', '.join(self.list_strategies())
            raise KeyError(f"Strategy '{name}' not found. Available strategies: {available}")
        
        return self._strategies[name]
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def get_all_strategies(self) -> Dict[str, BaseTextStrategy]:
        """Get all registered strategies.
        
        Returns:
            Dictionary mapping strategy names to instances
        """
        return self._strategies.copy()


# Global registry instance
_registry = StrategyRegistry()


def get_strategy(name: str) -> BaseTextStrategy:
    """Get a strategy by name from the global registry.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy instance
        
    Raises:
        KeyError: If strategy not found
    """
    return _registry.get_strategy(name)


def list_strategies() -> List[str]:
    """List all available strategy names from the global registry.
    
    Returns:
        List of strategy names
    """
    return _registry.list_strategies()


def get_all_strategies() -> Dict[str, BaseTextStrategy]:
    """Get all registered strategies from the global registry.
    
    Returns:
        Dictionary mapping strategy names to instances
    """
    return _registry.get_all_strategies()


def register_strategy(name: str, strategy: BaseTextStrategy):
    """Register a custom strategy with the global registry.
    
    Args:
        name: Strategy name
        strategy: Strategy instance
    """
    _registry.register(name, strategy)
