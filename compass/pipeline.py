"""Pipeline implementation for Compass."""

from typing import Dict, Any, Set, Optional, List, Tuple, Callable
import inspect


class Component:
    """
    Base class for all Compass components.
    
    All pipeline components should inherit from this class and implement the run method.
    """
    
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the component with the given inputs.
        
        Args:
            **kwargs: Input parameters for the component.
            
        Returns:
            Dict[str, Any]: Output of the component.
        """
        raise NotImplementedError("Component subclasses must implement run()")


class Pipeline:
    """
    Pipeline for connecting and running components in sequence.
    
    A pipeline connects components together and handles passing outputs from one component
    to the inputs of other components based on the pipeline topology.
    """
    
    def __init__(self) -> None:
        """Initialize an empty pipeline."""
        self.components: Dict[str, Component] = {}
        self.connections: Dict[str, Dict[str, str]] = {}
        self.root_components: Set[str] = set()
    
    def add_component(self, name: str, component: Component) -> None:
        """
        Add a component to the pipeline.
        
        Args:
            name: Unique name for the component.
            component: Component instance to add.
        
        Raises:
            ValueError: If a component with the same name already exists.
        """
        if name in self.components:
            raise ValueError(f"Component '{name}' already exists in the pipeline")
        
        self.components[name] = component
        self.connections[name] = {}
        self.root_components.add(name)
    
    def connect(self, from_component: str, to_component: str) -> None:
        """
        Connect two components in the pipeline.
        
        Args:
            from_component: Name of the source component.
            to_component: Name of the destination component.
            
        Raises:
            ValueError: If any of the components doesn't exist.
        """
        if from_component not in self.components:
            raise ValueError(f"Component '{from_component}' does not exist")
        if to_component not in self.components:
            raise ValueError(f"Component '{to_component}' does not exist")
        
        self.connections[from_component][to_component] = to_component
        if to_component in self.root_components:
            self.root_components.remove(to_component)
    
    def _get_run_order(self) -> List[str]:
        """
        Determine the order in which components should be run.
        
        Returns:
            List[str]: Ordered list of component names.
        """
        visited: Set[str] = set()
        order: List[str] = []
        
        def dfs(node: str) -> None:
            visited.add(node)
            for next_node in self.connections[node]:
                if next_node not in visited:
                    dfs(next_node)
            order.append(node)
        
        # Start DFS from each root component
        for root in self.root_components:
            if root not in visited:
                dfs(root)
        
        # Reverse the order to get correct execution sequence
        return order[::-1]
    
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the pipeline with the given inputs.
        
        Args:
            **kwargs: Input parameters for the pipeline.
            
        Returns:
            Dict[str, Any]: Results from all components.
        """
        results: Dict[str, Dict[str, Any]] = {}
        run_order = self._get_run_order()
        
        # Process each component in order
        for component_name in run_order:
            component = self.components[component_name]
            
            # Gather inputs for this component
            component_inputs = {}
            
            # First add any direct pipeline inputs that match the component's parameters
            component_params = inspect.signature(component.run).parameters
            for param_name in component_params:
                if param_name in kwargs:
                    component_inputs[param_name] = kwargs[param_name]
            
            # Then add outputs from previous components
            for prev_component, connections in self.connections.items():
                if component_name in connections and prev_component in results:
                    # Add all outputs from the previous component
                    component_inputs.update(results[prev_component])
            
            # Run the component
            component_output = component.run(**component_inputs)
            results[component_name] = component_output
        
        # Flatten results by combining all components' outputs
        flat_results = {}
        for component_results in results.values():
            flat_results.update(component_results)
        
        return flat_results