"""
BeliefBarChartSystem - A Drake LeafSystem for visualizing belief as a bar chart.

This system renders a 3-bar chart in Meshcat based on belief probability vector
received from BeliefEstimatorSystem. It implements the visualization layer
in the Perception -> Estimation -> Visualization pipeline for the discrete
3-bin Bayes filter.

The system is purely for visualization - it contains no estimation logic.
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    Box,
    Rgba,
    Meshcat,
    RigidTransform,
)


class BeliefBarChartSystem(LeafSystem):
    """
    A Drake System that visualizes belief as a bar chart in Meshcat.
    
    The system:
    - Receives belief vector from BeliefEstimatorSystem
    - Renders 3 bars with heights proportional to belief probabilities
    - Updates visualization during Publish events
    
    This system is stateless (no discrete state) - it only visualizes
    what it receives from upstream estimation systems.
    
    Inputs:
        belief (n_buckets D): Probability vector [P(A), P(B), P(C)]
    """
    
    # Bar colors for each bucket (colorblind-friendly palette)
    COLORS = [
        Rgba(0.2, 0.6, 0.9, 0.8),   # Bucket A: Blue
        Rgba(0.9, 0.4, 0.3, 0.8),   # Bucket B: Red/Coral
        Rgba(0.3, 0.8, 0.4, 0.8),   # Bucket C: Green
    ]
    
    BUCKET_LABELS = ['A', 'B', 'C']
    
    def __init__(
        self,
        meshcat: Meshcat,
        n_buckets: int = 3,
        chart_position: np.ndarray = None,
        bar_width: float = 0.08,
        bar_spacing: float = 0.12,
        max_height: float = 0.5,
        publish_period: float = 0.02,
    ):
        """
        Args:
            meshcat: Meshcat instance for visualization
            n_buckets: Number of discrete hypothesis buckets (default: 3)
            chart_position: Position of the chart in world frame [x, y, z]
            bar_width: Width of each bar (meters)
            bar_spacing: Spacing between bar centers (meters)
            max_height: Maximum bar height when P=1.0 (meters)
            publish_period: Meshcat publish period in seconds
        """
        LeafSystem.__init__(self)
        
        self._meshcat: Meshcat | None = meshcat
        self._n_buckets = n_buckets
        self._bar_width = bar_width
        self._bar_spacing = bar_spacing
        self._max_height = max_height
        
        # Default chart position (above and behind the workspace)
        if chart_position is None:
            chart_position = np.array([-0.3, 0.5, 0.8])
        self._chart_position = chart_position
        
        # Input port: belief vector from BeliefEstimatorSystem
        self._belief_port = self.DeclareVectorInputPort("belief", n_buckets)
        
        # Periodic publish for Meshcat visualization
        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period,
            offset_sec=0.0,
            publish=self._DoPublishBarChart,
        )
        
        # Setup Meshcat visualization objects
        if meshcat is not None:
            self._setup_meshcat_objects()
    
    def _setup_meshcat_objects(self):
        """Create initial Meshcat objects for the bar chart."""
        # Create base plate for the chart
        base_width = self._bar_spacing * (self._n_buckets + 0.5)
        base_depth = self._bar_width * 1.5
        base_height = 0.01
        
        self._meshcat.SetObject(
            "belief_chart/base",
            Box(base_width, base_depth, base_height),
            Rgba(0.3, 0.3, 0.3, 0.5)
        )
        self._meshcat.SetTransform(
            "belief_chart/base",
            RigidTransform(self._chart_position)
        )
        
        # Create initial bars (height will be updated each frame)
        for i in range(self._n_buckets):
            bar_x = self._chart_position[0] + (i - (self._n_buckets - 1) / 2) * self._bar_spacing
            bar_y = self._chart_position[1]
            bar_z = self._chart_position[2]
            
            # Initial bar with small height
            initial_height = 0.02
            color = self.COLORS[i] if i < len(self.COLORS) else Rgba(0.5, 0.5, 0.5, 0.8)
            
            self._meshcat.SetObject(
                f"belief_chart/bar_{i}",
                Box(self._bar_width, self._bar_width, initial_height),
                color
            )
            self._meshcat.SetTransform(
                f"belief_chart/bar_{i}",
                RigidTransform([bar_x, bar_y, bar_z + initial_height / 2])
            )
    
    def _DoPublishBarChart(self, context):
        """
        Publish the bar chart to Meshcat.
        
        Reads belief from input port and updates bar heights accordingly.
        """
        if self._meshcat is None:
            return
        
        # Get belief from input port (from BeliefEstimatorSystem)
        belief = self._belief_port.Eval(context)
        
        current_time = context.get_time()
        
        # Update each bar
        for i in range(self._n_buckets):
            prob = belief[i]
            
            # Compute bar height (minimum height for visibility)
            bar_height = max(0.01, prob * self._max_height)
            
            # Compute bar position (centered at chart_position, offset by index)
            bar_x = self._chart_position[0] + (i - (self._n_buckets - 1) / 2) * self._bar_spacing
            bar_y = self._chart_position[1]
            bar_z = self._chart_position[2] + bar_height / 2
            
            # Update bar shape with new height
            color = self.COLORS[i] if i < len(self.COLORS) else Rgba(0.5, 0.5, 0.5, 0.8)
            self._meshcat.SetObject(
                f"belief_chart/bar_{i}",
                Box(self._bar_width, self._bar_width, bar_height),
                color
            )
            
            # Update bar transform
            self._meshcat.SetTransform(
                f"belief_chart/bar_{i}",
                RigidTransform([bar_x, bar_y, bar_z]),
                time_in_recording=current_time
            )

