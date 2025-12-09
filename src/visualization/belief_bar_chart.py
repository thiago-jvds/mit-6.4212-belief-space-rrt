"""
BeliefBarChartSystem - A Drake LeafSystem for visualizing belief as bars near bins.

This system renders one bar per bin hypothesis in Meshcat, with each bar positioned
near its corresponding bin. The bar heights are proportional to the belief probabilities
received from BeliefEstimatorSystem.

Bar positions are computed from bin transforms:
- Bar 0 position = X_W_bin0 * X_bin0_chart (shows P(bin0), located near bin0)
- Bar 1 position = X_W_bin1 * X_bin1_chart (shows P(bin1), located near bin1)

Bar colors change dynamically based on probability:
- 0.0 → Red (low probability)
- 0.5 → Yellow (uncertain)
- 1.0 → Green (high probability)

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
    A Drake System that visualizes belief as bars in Meshcat.
    
    The system:
    - Receives belief vector from BeliefEstimatorSystem
    - Renders one bar per bin, positioned near its respective bin
    - Bar heights are proportional to belief probabilities
    - Bar colors indicate probability: red (0) → yellow (0.5) → green (1)
    - Updates visualization during Publish events
    
    This system is stateless (no discrete state) - it only visualizes
    what it receives from upstream estimation systems.
    
    Inputs:
        belief (n_bins D): Probability vector [P(bin0), P(bin1)]
    """
    
    # Alpha for bar transparency
    BAR_ALPHA = 0.85
    
    def __init__(
        self,
        meshcat: Meshcat,
        n_bins: int = 2,
        X_W_bin0: RigidTransform = None,
        X_W_bin1: RigidTransform = None,
        X_bin0_chart: RigidTransform = None,
        X_bin1_chart: RigidTransform = None,
        bar_width: float = 0.06,
        max_height: float = 0.2,
        publish_period: float = 0.02,
    ):
        """
        Args:
            meshcat: Meshcat instance for visualization
            n_bins: Number of discrete hypothesis bins (default: 2)
            X_W_bin0: World-to-bin0 transform
            X_W_bin1: World-to-bin1 transform
            X_bin0_chart: Relative transform from bin0 frame to bar position
            X_bin1_chart: Relative transform from bin1 frame to bar position
            bar_width: Width of each bar (meters)
            max_height: Maximum bar height when P=1.0 (meters)
            publish_period: Meshcat publish period in seconds
        """
        LeafSystem.__init__(self)
        
        self._meshcat: Meshcat | None = meshcat
        self._n_bins = n_bins
        self._bar_width = bar_width
        self._max_height = max_height
        
        # Compute world position for each bar (one bar per bin)
        # Bar i is positioned near bin i with its own relative transform
        self._bar_positions = []
        
        bin_transforms = [X_W_bin0, X_W_bin1]
        chart_offsets = [X_bin0_chart, X_bin1_chart]
        
        for i in range(n_bins):
            # Apply bin-specific relative transform from bin frame to bar position
            if bin_transforms[i] is not None and chart_offsets[i] is not None:
                X_W_bar = bin_transforms[i].multiply(chart_offsets[i])
                self._bar_positions.append(X_W_bar.translation())
            else:
                raise ValueError(f"No transform provided for bin {i}")
        
        # Input port: belief vector from BeliefEstimatorSystem
        self._belief_port = self.DeclareVectorInputPort("belief", n_bins)
        
        # Periodic publish for Meshcat visualization
        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period,
            offset_sec=0.0,
            publish=self._DoPublishBarChart,
        )
        
        # Setup Meshcat visualization objects
        if meshcat is not None:
            self._setup_meshcat_objects()
    
    @staticmethod
    def _probability_to_color(prob: float, alpha: float = 0.85) -> Rgba:
        """
        Convert probability to color: red (0) → yellow (0.5) → green (1).
        
        Args:
            prob: Probability value in [0, 1]
            alpha: Transparency value
            
        Returns:
            Rgba color
        """
        prob = np.clip(prob, 0.0, 1.0)
        
        if prob <= 0.5:
            # Red to Yellow: R=1, G increases from 0 to 1
            r = 1.0
            g = prob * 2.0  # 0 at prob=0, 1 at prob=0.5
            b = 0.0
        else:
            # Yellow to Green: R decreases from 1 to 0, G=1
            r = 1.0 - (prob - 0.5) * 2.0  # 1 at prob=0.5, 0 at prob=1
            g = 1.0
            b = 0.0
        
        return Rgba(r, g, b, alpha)
    
    def _setup_meshcat_objects(self):
        """Create initial Meshcat objects for the bars (one per bin)."""
        for i in range(self._n_bins):
            bar_pos = self._bar_positions[i]
            
            # Initial bar with small height and yellow color (0.5 probability)
            initial_height = 0.02
            initial_color = self._probability_to_color(0.5, self.BAR_ALPHA)
            
            self._meshcat.SetObject(
                f"belief_chart/bar_{i}",
                Box(self._bar_width, self._bar_width, initial_height),
                initial_color
            )
            self._meshcat.SetTransform(
                f"belief_chart/bar_{i}",
                RigidTransform([bar_pos[0], bar_pos[1], bar_pos[2] + initial_height / 2])
            )
    
    def _DoPublishBarChart(self, context):
        """
        Publish the bars to Meshcat.
        
        Reads belief from input port and updates bar heights and colors.
        Each bar i shows belief[i] and is positioned near bin i.
        Color indicates probability: red (0) → yellow (0.5) → green (1).
        """
        if self._meshcat is None:
            return
        
        # Get belief from input port (from BeliefEstimatorSystem)
        belief = self._belief_port.Eval(context)
        
        current_time = context.get_time()
        
        # Update each bar (one bar per bin)
        for i in range(self._n_bins):
            prob = belief[i]
            bar_pos = self._bar_positions[i]
            
            # Compute bar height (minimum height for visibility)
            bar_height = max(0.01, prob * self._max_height)
            
            # Bar position: XY from bar_positions, Z raised by half the height
            bar_x = bar_pos[0]
            bar_y = bar_pos[1]
            bar_z = bar_pos[2] + bar_height / 2
            
            # Compute dynamic color based on probability
            color = self._probability_to_color(prob, self.BAR_ALPHA)
            
            # Update bar shape with new height and color
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
