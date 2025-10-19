# Bioinformatics Visualization Guide

## Overview

PathwayLens 2.0 provides comprehensive visualization capabilities for bioinformatics data analysis. This guide covers visualization types, best practices, and implementation details for creating effective biological visualizations.

## Visualization Types

### 1. Pathway Enrichment Visualizations

#### Dot Plot

**Purpose**: Display pathway enrichment results with significance and gene counts.

**Features**:
- X-axis: Gene ratio or enrichment score
- Y-axis: Pathway names
- Color: Significance level (p-value or adjusted p-value)
- Size: Number of genes in pathway
- Interactive tooltips with detailed information

**Use Cases**:
- ORA (Over-Representation Analysis) results
- GSEA (Gene Set Enrichment Analysis) results
- Multi-database comparison

**Implementation**:
```python
def create_dot_plot(data, title="Pathway Enrichment"):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['gene_ratio'],
        y=data['pathway_name'],
        mode='markers',
        marker=dict(
            size=data['gene_count'],
            color=data['p_value'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="P-value")
        ),
        text=data['pathway_description'],
        hovertemplate='<b>%{y}</b><br>' +
                     'Gene Ratio: %{x}<br>' +
                     'P-value: %{marker.color}<br>' +
                     'Genes: %{marker.size}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Gene Ratio",
        yaxis_title="Pathway",
        height=600
    )
    
    return fig
```

#### Volcano Plot

**Purpose**: Visualize significance vs. effect size for pathway analysis.

**Features**:
- X-axis: Effect size (log2 fold change, enrichment score)
- Y-axis: Significance (-log10 p-value)
- Color: Significance categories
- Interactive selection and filtering

**Use Cases**:
- Differential pathway analysis
- GSEA results visualization
- Multi-condition comparison

**Implementation**:
```python
def create_volcano_plot(data, title="Volcano Plot"):
    fig = go.Figure()
    
    # Color points based on significance
    colors = ['red' if p < 0.01 else 'blue' if p < 0.05 else 'gray' 
              for p in data['p_value']]
    
    fig.add_trace(go.Scatter(
        x=data['effect_size'],
        y=-np.log10(data['p_value']),
        mode='markers',
        marker=dict(color=colors, size=8),
        text=data['pathway_name'],
        hovertemplate='<b>%{text}</b><br>' +
                     'Effect Size: %{x}<br>' +
                     'P-value: %{y}<br>' +
                     '<extra></extra>'
    ))
    
    # Add significance lines
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", 
                  annotation_text="p = 0.05")
    fig.add_hline(y=-np.log10(0.01), line_dash="dash", 
                  annotation_text="p = 0.01")
    
    fig.update_layout(
        title=title,
        xaxis_title="Effect Size",
        yaxis_title="-log10(P-value)",
        height=600
    )
    
    return fig
```

### 2. Network Visualizations

#### Pathway Network

**Purpose**: Display pathway interactions and gene relationships.

**Features**:
- Nodes: Pathways or genes
- Edges: Interactions or relationships
- Node size: Pathway size or significance
- Edge thickness: Interaction strength
- Interactive zoom and pan

**Use Cases**:
- Pathway interaction analysis
- Gene network visualization
- Multi-pathway comparison

**Implementation**:
```python
def create_pathway_network(data, title="Pathway Network"):
    # Create network graph
    G = nx.Graph()
    
    # Add nodes and edges
    for pathway in data['pathways']:
        G.add_node(pathway['id'], 
                  name=pathway['name'],
                  size=pathway['gene_count'],
                  p_value=pathway['p_value'])
    
    for interaction in data['interactions']:
        G.add_edge(interaction['source'], 
                  interaction['target'],
                  weight=interaction['strength'])
    
    # Create Plotly network
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['name'])
        node_size.append(G.nodes[node]['size'])
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           text=node_text,
                           marker=dict(size=node_size,
                                     color=node_size,
                                     colorscale='Viridis',
                                     showscale=True))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(title=title,
                                  showlegend=False,
                                  hovermode='closest',
                                  margin=dict(b=20,l=5,r=5,t=40),
                                  annotations=[ dict(
                                      text="Pathway Network",
                                      showarrow=False,
                                      xref="paper", yref="paper",
                                      x=0.005, y=-0.002,
                                      xanchor='left', yanchor='bottom',
                                      font=dict(color="black", size=12)
                                  )],
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig
```

### 3. Multi-Dataset Comparison

#### Heatmap

**Purpose**: Compare pathway enrichment across multiple conditions.

**Features**:
- Rows: Pathways
- Columns: Conditions/datasets
- Color: Enrichment score or significance
- Interactive clustering and filtering

**Use Cases**:
- Multi-condition pathway analysis
- Time-series pathway analysis
- Cross-species comparison

**Implementation**:
```python
def create_heatmap(data, title="Pathway Comparison Heatmap"):
    fig = go.Figure(data=go.Heatmap(
        z=data['enrichment_scores'],
        x=data['conditions'],
        y=data['pathways'],
        colorscale='RdBu',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>' +
                     'Condition: %{x}<br>' +
                     'Enrichment: %{z}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Conditions",
        yaxis_title="Pathways",
        height=800
    )
    
    return fig
```

#### Venn Diagram

**Purpose**: Show pathway overlap between datasets.

**Features**:
- Circles: Datasets
- Overlaps: Shared pathways
- Interactive tooltips with pathway details

**Use Cases**:
- Pathway overlap analysis
- Multi-dataset comparison
- Venn diagram visualization

**Implementation**:
```python
def create_venn_diagram(data, title="Pathway Overlap"):
    # Use matplotlib-venn for Venn diagram
    from matplotlib_venn import venn3, venn2
    
    if len(data['datasets']) == 2:
        venn2(subsets=data['overlaps'], 
              set_labels=data['dataset_names'])
    elif len(data['datasets']) == 3:
        venn3(subsets=data['overlaps'], 
              set_labels=data['dataset_names'])
    
    plt.title(title)
    plt.show()
```

### 4. Multi-Omics Visualizations

#### Multi-Omics Heatmap

**Purpose**: Integrate multiple omics data types in a single visualization.

**Features**:
- Rows: Genes or pathways
- Columns: Omics data types
- Color: Expression or enrichment values
- Interactive filtering and clustering

**Use Cases**:
- Multi-omics integration
- Cross-omics pathway analysis
- Comprehensive data visualization

**Implementation**:
```python
def create_multi_omics_heatmap(data, title="Multi-Omics Heatmap"):
    fig = go.Figure(data=go.Heatmap(
        z=data['values'],
        x=data['omics_types'],
        y=data['genes'],
        colorscale='RdBu',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>' +
                     'Omics Type: %{x}<br>' +
                     'Value: %{z}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Omics Data Types",
        yaxis_title="Genes",
        height=800
    )
    
    return fig
```

#### Sankey Diagram

**Purpose**: Show data flow between omics layers and pathways.

**Features**:
- Nodes: Omics layers, pathways, genes
- Links: Data flow and relationships
- Interactive exploration

**Use Cases**:
- Multi-omics data flow
- Pathway-gene relationships
- Data integration visualization

**Implementation**:
```python
def create_sankey_diagram(data, title="Multi-Omics Data Flow"):
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=data['node_labels'],
            color=data['node_colors']
        ),
        link=dict(
            source=data['link_sources'],
            target=data['link_targets'],
            value=data['link_values'],
            color=data['link_colors']
        )
    )])
    
    fig.update_layout(title_text=title, font_size=10)
    
    return fig
```

## Best Practices

### 1. Color Schemes

#### Biological Color Schemes

- **Red-Blue**: Up-regulation (red) vs. down-regulation (blue)
- **Green-Red**: Positive (green) vs. negative (red) effects
- **Viridis**: Perceptually uniform color scale
- **Colorblind-Friendly**: Use colorblind-friendly palettes

#### Implementation

```python
# Colorblind-friendly palette
COLORS = {
    'red': '#E31A1C',
    'blue': '#1F78B4',
    'green': '#33A02C',
    'orange': '#FF7F00',
    'purple': '#6A3D9A',
    'brown': '#B15928'
}

# Significance color mapping
def get_significance_color(p_value):
    if p_value < 0.001:
        return '#E31A1C'  # Red
    elif p_value < 0.01:
        return '#FF7F00'  # Orange
    elif p_value < 0.05:
        return '#1F78B4'  # Blue
    else:
        return '#808080'  # Gray
```

### 2. Interactive Features

#### Tooltips

```python
def create_tooltip(data):
    return '<b>%{text}</b><br>' + \
           'P-value: %{customdata[0]:.3f}<br>' + \
           'Genes: %{customdata[1]}<br>' + \
           'Database: %{customdata[2]}<br>' + \
           '<extra></extra>'
```

#### Zoom and Pan

```python
def enable_zoom_pan(fig):
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(fixedrange=False),
        dragmode='pan'
    )
```

#### Selection and Filtering

```python
def add_selection_filter(fig, data):
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="All",
                         method="restyle",
                         args=["visible", [True] * len(data)]),
                    dict(label="Significant",
                         method="restyle",
                         args=["visible", [p < 0.05 for p in data['p_values']]])
                ]),
                direction="down",
                showactive=True,
            )
        ]
    )
```

### 3. Accessibility

#### Colorblind Accessibility

```python
# Use patterns and shapes in addition to color
def add_accessibility_features(fig):
    fig.update_traces(
        marker=dict(
            symbol=['circle', 'square', 'diamond', 'triangle-up'],
            line=dict(width=2, color='black')
        )
    )
```

#### Screen Reader Support

```python
def add_aria_labels(fig):
    fig.update_layout(
        title=dict(
            text="Pathway Enrichment Analysis",
            xref="paper",
            yref="paper"
        ),
        annotations=[
            dict(
                text="Interactive pathway enrichment visualization",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                xanchor="center", yanchor="top"
            )
        ]
    )
```

### 4. Performance Optimization

#### Large Dataset Handling

```python
def optimize_large_dataset(data, max_points=1000):
    if len(data) > max_points:
        # Sample data for visualization
        data = data.sample(n=max_points)
        # Add note about sampling
        return data, f"Showing {max_points} of {len(data)} pathways"
    return data, None
```

#### Lazy Loading

```python
def lazy_load_visualization(data_id):
    # Load data asynchronously
    import asyncio
    
    async def load_data():
        # Simulate async data loading
        await asyncio.sleep(0.1)
        return load_pathway_data(data_id)
    
    return asyncio.create_task(load_data())
```

## Implementation Guidelines

### 1. Modular Design

```python
class VisualizationEngine:
    def __init__(self):
        self.plotters = {
            'dot_plot': self.create_dot_plot,
            'volcano_plot': self.create_volcano_plot,
            'heatmap': self.create_heatmap,
            'network': self.create_network
        }
    
    def create_visualization(self, plot_type, data, **kwargs):
        if plot_type not in self.plotters:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        return self.plotters[plot_type](data, **kwargs)
```

### 2. Configuration Management

```python
class VisualizationConfig:
    def __init__(self):
        self.defaults = {
            'width': 800,
            'height': 600,
            'colorscale': 'Viridis',
            'font_size': 12,
            'title_size': 16
        }
    
    def get_config(self, plot_type, custom_config=None):
        config = self.defaults.copy()
        if custom_config:
            config.update(custom_config)
        return config
```

### 3. Error Handling

```python
def safe_visualization(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return create_error_plot(str(e))
    return wrapper

@safe_visualization
def create_dot_plot(data, **kwargs):
    # Visualization code
    pass
```

### 4. Testing

```python
def test_visualization():
    # Test data
    test_data = {
        'pathways': ['Pathway A', 'Pathway B'],
        'p_values': [0.01, 0.05],
        'gene_counts': [10, 20]
    }
    
    # Create visualization
    fig = create_dot_plot(test_data)
    
    # Assertions
    assert fig is not None
    assert len(fig.data) > 0
    assert fig.layout.title.text == "Pathway Enrichment"
```

## Export and Sharing

### 1. Export Formats

```python
def export_visualization(fig, format='html', filename=None):
    if format == 'html':
        return fig.to_html(filename)
    elif format == 'png':
        return fig.to_image(format='png', filename=filename)
    elif format == 'svg':
        return fig.to_image(format='svg', filename=filename)
    elif format == 'pdf':
        return fig.to_image(format='pdf', filename=filename)
```

### 2. Interactive Sharing

```python
def share_visualization(fig, title="Pathway Analysis"):
    # Create shareable link
    import plotly.io as pio
    
    html_string = pio.to_html(fig, include_plotlyjs=True)
    
    # Save to file or upload to cloud
    with open('shared_plot.html', 'w') as f:
        f.write(html_string)
    
    return 'shared_plot.html'
```

## Future Enhancements

### 1. 3D Visualizations

- 3D pathway networks
- 3D multi-omics heatmaps
- Virtual reality support

### 2. Advanced Interactivity

- Real-time data updates
- Collaborative editing
- Advanced filtering and search

### 3. Machine Learning Integration

- Automated visualization selection
- Pattern recognition
- Predictive visualizations

### 4. Mobile Support

- Responsive design
- Touch interactions
- Mobile-optimized layouts

## Resources

### Documentation

- [Plotly.js Documentation](https://plotly.com/javascript/)
- [D3.js Documentation](https://d3js.org/)
- [Bioinformatics Visualization Best Practices](https://example.com)

### Tools and Libraries

- **Plotly.js**: Interactive plotting library
- **D3.js**: Data-driven document manipulation
- **Cytoscape.js**: Graph visualization
- **Chart.js**: Simple charting library
- **Observable**: Data visualization platform

### Examples

- [PathwayLens Examples](https://github.com/pathwaylens/examples)
- [Bioinformatics Visualization Gallery](https://example.com/gallery)
- [Interactive Demos](https://example.com/demos)
