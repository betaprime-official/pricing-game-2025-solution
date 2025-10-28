import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def loss_ratio_chart(grouped_df: pd.DataFrame, var_name: str) -> go.Figure:
    """Create Loss Ratio chart with Average Premium on secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Average Premium bars on PRIMARY axis (will be in background)
    fig.add_trace(
        go.Bar(x=grouped_df[var_name], y=grouped_df['avg_premium'], 
               name='Avg Premium', marker_color='#A0E8AF', opacity=0.7),
        secondary_y=False
    )
    
    # Loss Ratio line on SECONDARY axis (will be in foreground)
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['loss_ratio'], 
                   name='Loss Ratio (%)', mode='lines+markers',
                   line=dict(color='#2E86AB', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text="Average Premium ($)", secondary_y=False)
    fig.update_yaxes(title_text="Loss Ratio (%)", secondary_y=True)
    fig.update_layout(
        title=f"Loss Ratio and Average Premium by {var_name}", 
        hovermode='x unified', 
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis2=dict(
            showgrid=False
        ),
        plot_bgcolor='white'
    )
    return fig


def pure_premium_chart(grouped_df: pd.DataFrame, var_name: str) -> go.Figure:
    """Create Pure Premium chart with Exposures on secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Exposures bars on PRIMARY axis (will be in background)
    fig.add_trace(
        go.Bar(x=grouped_df[var_name], y=grouped_df['exposure'], 
               name='Exposures', marker_color='#C7D3E3', opacity=0.7),
        secondary_y=False
    )
    
    # Pure Premium line on SECONDARY axis (will be in foreground)
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['pure_premium'], 
                   name='Pure Premium ($)', mode='lines+markers',
                   line=dict(color='#F18F01', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text="Exposures", secondary_y=False)
    fig.update_yaxes(title_text="Pure Premium ($)", secondary_y=True)
    fig.update_layout(
        title=f"Pure Premium and Exposures by {var_name}", 
        hovermode='x unified', 
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis2=dict(
            showgrid=False
        ),
        plot_bgcolor='white'
    )
    return fig


def frequency_chart(grouped_df: pd.DataFrame, var_name: str) -> go.Figure:
    """Create Frequency chart with Exposures on secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Exposures bars on PRIMARY axis (will be in background)
    fig.add_trace(
        go.Bar(x=grouped_df[var_name], y=grouped_df['exposure'], 
               name='Exposures', marker_color='#E8D5C4', opacity=0.7),
        secondary_y=False
    )
    
    # Frequency line on SECONDARY axis (will be in foreground)
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['frequency'], 
                   name='Frequency', mode='lines+markers',
                   line=dict(color='#8B5A3C', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text="Exposures", secondary_y=False)
    fig.update_yaxes(title_text="Claims per Exposure", secondary_y=True)
    fig.update_layout(
        title=f"Frequency by {var_name}", 
        hovermode='x unified', 
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis2=dict(
            showgrid=False
        ),
        plot_bgcolor='white'
    )
    return fig


def severity_chart(grouped_df: pd.DataFrame, var_name: str) -> go.Figure:
    """Create Severity chart with Claim Counts on secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Claim Counts bars on PRIMARY axis (will be in background)
    fig.add_trace(
        go.Bar(x=grouped_df[var_name], y=grouped_df['claim_counts'], 
               name='Claim Counts', marker_color='#F5E6D3', opacity=0.7),
        secondary_y=False
    )
    
    # Severity line on SECONDARY axis (will be in foreground)
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['severity'], 
                   name='Severity ($)', mode='lines+markers',
                   line=dict(color='#D4A574', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text="Claim Counts", secondary_y=False)
    fig.update_yaxes(title_text="Avg Loss per Claim ($)", secondary_y=True)
    fig.update_layout(
        title=f"Severity by {var_name}", 
        hovermode='x unified', 
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis2=dict(
            showgrid=False
        ),
        plot_bgcolor='white'
    )
    return fig


def create_univariate_charts(grouped_df: pd.DataFrame, var_name: str) -> dict:
    """
    Create all univariate analysis charts at once.
    Returns a dictionary with all chart figures.
    """
    return {
        'loss_ratio': loss_ratio_chart(grouped_df, var_name),
        'pure_premium': pure_premium_chart(grouped_df, var_name),
        'frequency': frequency_chart(grouped_df, var_name),
        'severity': severity_chart(grouped_df, var_name)
    }


def relativities_chart(params: pd.Series, ci_lower: pd.Series, ci_upper: pd.Series, 
                       var_name: str, base_level: str = None) -> go.Figure:
    """Create chart showing model relativities with confidence intervals."""
    # Filter to just the variable of interest (exclude Intercept)
    # Handle both simple C(var) and C(var, Treatment(...)) patterns
    # Use more flexible pattern that matches C(var_name followed by either ) or ,
    pattern = f'C\\({var_name}[,\\)]'
    var_params = params[params.index.str.contains(pattern, regex=True)]
    var_ci_lower = ci_lower[ci_lower.index.str.contains(pattern, regex=True)]
    var_ci_upper = ci_upper[ci_upper.index.str.contains(pattern, regex=True)]
    
    # Check if we found any parameters for this variable
    if len(var_params) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title=f"No parameters found for variable: {var_name}",
            xaxis_title=var_name,
            yaxis_title="Relativity",
            template='plotly_white',
            annotations=[
                dict(
                    text=f"Variable '{var_name}' not found in model parameters",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )
        return fig
    
    # Clean up labels - handle both patterns
    labels = []
    for label in var_params.index:
        if '[T.' in label:
            # Extract the level from patterns like C(var, Treatment(...))[T.level] or C(var)[T.level]
            clean_label = label.split('[T.')[1].replace(']', '')
            labels.append(clean_label)
        else:
            # Handle other patterns if needed
            labels.append(label)
    
    # Add base level if provided
    if base_level:
        labels = [base_level] + labels
        var_params = pd.concat([pd.Series([1.0], index=[base_level]), var_params])
        var_ci_lower = pd.concat([pd.Series([1.0], index=[base_level]), var_ci_lower])
        var_ci_upper = pd.concat([pd.Series([1.0], index=[base_level]), var_ci_upper])
    
    fig = go.Figure()
    
    # Add relativities with error bars
    fig.add_trace(go.Scatter(
        x=labels,
        y=var_params.values,
        error_y=dict(
            type='data',
            symmetric=False,
            array=var_ci_upper.values - var_params.values,
            arrayminus=var_params.values - var_ci_lower.values
        ),
        mode='markers',
        marker=dict(size=10, color='#2E86AB'),
        name='Relativities'
    ))
    
    # Add reference line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Model Relativities for {var_name}",
        xaxis_title=var_name,
        yaxis_title="Relativity",
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def rating_relativities_chart(rating_json: dict, var_name: str) -> go.Figure:
    """Create chart showing relativities from rating JSON structure."""
    if 'relativities' not in rating_json or var_name not in rating_json['relativities']:
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"No relativities found for {var_name}",
            xaxis_title=var_name,
            yaxis_title="Relativity",
            template='plotly_white'
        )
        return fig
    
    # Get relativities for this variable
    var_relativities = rating_json['relativities'][var_name]
    
    # Convert to lists for plotting
    levels = list(var_relativities.keys())
    relativities = list(var_relativities.values())
    
    # Create colors - highlight base level (1.0) in different color
    colors = ['#F18F01' if rel == 1.0 else '#2E86AB' for rel in relativities]
    
    fig = go.Figure()
    
    # Add relativities as bars
    fig.add_trace(go.Bar(
        x=levels,
        y=relativities,
        marker_color=colors,
        name='Relativities',
        text=[f'{rel:.4f}' for rel in relativities],
        textposition='outside'
    ))
    
    # Add reference line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Rating Relativities for {var_name}",
        xaxis_title=var_name,
        yaxis_title="Relativity",
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def observed_vs_predicted_chart(grouped_df: pd.DataFrame, var_name: str, 
                                offset_name: str = 'Exposure') -> go.Figure:
    """Create chart comparing observed vs predicted rates with offset as bars."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Offset (exposure or claim counts) as bars on PRIMARY axis
    fig.add_trace(
        go.Bar(x=grouped_df[var_name], y=grouped_df['offset_sum'], 
               name=offset_name, marker_color='#C7D3E3', opacity=0.5),
        secondary_y=False
    )
    
    # Observed rates as line on SECONDARY axis
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['observed_rate'], 
                   name='Observed', mode='lines+markers',
                   line=dict(color='#2E86AB', width=2),
                   marker=dict(size=8)),
        secondary_y=True
    )
    
    # Predicted rates as line on SECONDARY axis
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['predicted_rate'], 
                   name='Predicted', mode='lines+markers',
                   line=dict(color='#F18F01', width=2, dash='dash'),
                   marker=dict(size=8)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text=offset_name, secondary_y=False)
    fig.update_yaxes(title_text="Rate", secondary_y=True)
    
    fig.update_layout(
        title=f"Observed vs Predicted Rates by {var_name}",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def premium_delta_histogram(df: pd.DataFrame, delta_col: str = 'premium_delta') -> go.Figure:
    """Create histogram of premium changes with reference lines."""
    fig = px.histogram(
        df, 
        x=delta_col, 
        nbins=50,
        title='Distribution of Premium Changes (%)',
        labels={delta_col: 'Premium Change (%)', 'count': 'Number of Policies'},
        color_discrete_sequence=['lightblue']
    )

    # Add vertical line at 0 (no change)
    fig.add_vline(x=0, line_dash="dash", line_color="red")

    # Add overall change line
    overall_change = ((df['new_comm_premium'].sum() / df['commercial_premium'].sum()) - 1.0) * 100
    fig.add_vline(x=overall_change, line_dash="dot", line_color="green", 
                  annotation_text=f"Overall Change: {overall_change:.1f}%")

    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    return fig


def retention_analysis_chart(retention_by_var: pd.DataFrame, variable: str) -> go.Figure:
    """Create retention analysis chart with premium change bars and retention rate line."""
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    
    # Bar chart for average premium delta (primary y-axis)
    fig.add_trace(
        go.Bar(
            x=retention_by_var[variable],
            y=retention_by_var['avg_premium_delta'],
            name="Avg Premium Change (%)",
            marker_color='lightblue',
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # Line chart for retention rate (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=retention_by_var[variable],
            y=retention_by_var['retention_rate'],
            mode='lines+markers',
            name="Retention Rate",
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text="Average Premium Change (%)", secondary_y=False)
    fig.update_yaxes(title_text="Retention Rate", secondary_y=True, tickformat='.1%', range=[0, 1])
    
    fig.update_layout(
        title=f"Retention Rate and Premium Change by {variable}",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def profitability_comparison_chart(grouped_df: pd.DataFrame, var_name: str) -> go.Figure:
    """Create profitability comparison chart showing old vs new loss ratios (following loss_ratio_chart pattern)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Average Premium bars on PRIMARY axis (will be in background)
    fig.add_trace(
        go.Bar(x=grouped_df[var_name], y=grouped_df['avg_premium_old'], 
               name='Avg Premium (Old)', marker_color='#A0E8AF', opacity=0.7),
        secondary_y=False
    )
    
    # Old Loss Ratio line on SECONDARY axis
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['loss_ratio_old'], 
                   name='Loss Ratio % (Old)', mode='lines+markers',
                   line=dict(color='#2E86AB', width=2)),
        secondary_y=True
    )
    
    # New Loss Ratio line on SECONDARY axis
    fig.add_trace(
        go.Scatter(x=grouped_df[var_name], y=grouped_df['loss_ratio_new'], 
                   name='Loss Ratio % (New)', mode='lines+markers',
                   line=dict(color='#F18F01', width=2)),
        secondary_y=True
    )
    
    # Add horizontal line at 100% (break-even) on secondary y-axis
    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=100, y1=100,
        xref="paper", yref="y2",  # y2 is the secondary y-axis
        line=dict(color="gray", width=2, dash="dash")
    )
    
    # Add annotation for break-even line
    fig.add_annotation(
        x=0.02, y=100,
        xref="paper", yref="y2",
        text="Break-even (100%)",
        showarrow=False,
        font=dict(color="gray")
    )
    
    fig.update_xaxes(title_text=var_name)
    fig.update_yaxes(title_text="Average Premium ($)", secondary_y=False)
    fig.update_yaxes(title_text="Loss Ratio (%)", secondary_y=True)
    
    fig.update_layout(
        title=f"Expected Profitability Analysis: Old vs New Loss Ratios by {var_name}", 
        hovermode='x unified', 
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig