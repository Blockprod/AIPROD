import React, { useState, useEffect } from 'react';
import '../styles/CostEstimator.css';

/**
 * CostEstimator Component
 * Real-time cost calculation display with breakdown visualization
 * 
 * Props:
 * - tier: Selected quality tier
 * - onCostUpdate: Callback when cost is calculated
 * - updateCostEstimate: Function from useQualityCalculator hook
 * - currentCost: Current cost breakdown from API
 * - loading: Loading state indicator
 */
const CostEstimator = ({
  tier = 'good',
  onCostUpdate,
  updateCostEstimate,
  currentCost = null,
  loading = false,
}) => {
  const [duration, setDuration] = useState(60);
  const [complexity, setComplexity] = useState('moderate');
  const [rushDelivery, setRushDelivery] = useState('standard');
  const [batchCount, setBatchCount] = useState(1);

  // Update cost estimate whenever inputs change
  useEffect(() => {
    if (updateCostEstimate) {
      updateCostEstimate({
        tier,
        duration_sec: duration,
        complexity,
        rush_delivery: rushDelivery,
        batch_count: batchCount,
      });
    }
  }, [tier, duration, complexity, rushDelivery, batchCount, updateCostEstimate]);

  const handleDurationChange = (e) => {
    const value = Math.max(1, parseInt(e.target.value) || 0);
    setDuration(value);
  };

  const handleComplexityChange = (e) => {
    setComplexity(e.target.value);
  };

  const handleRushDeliveryChange = (e) => {
    setRushDelivery(e.target.value);
  };

  const handleBatchCountChange = (e) => {
    const value = Math.max(1, parseInt(e.target.value) || 1);
    setBatchCount(value);
  };

  const getTierColor = (tierName) => {
    const colors = {
      good: '#2563eb',
      high: '#dc2626',
      ultra: '#a855f7',
    };
    return colors[tierName] || '#6b7280';
  };

  const getRushDeliveryTime = (rush) => {
    const times = {
      standard: 'Standard (30-120s)',
      express_6h: '6 Hour Express',
      express_2h: '2 Hour Express',
      on_demand: 'On-Demand (5 min)',
    };
    return times[rush] || 'Standard';
  };

  const getDurationMinutes = () => (duration / 60).toFixed(2);

  return (
    <div className="cost-estimator">
      <div className="estimator-header">
        <h2>💰 Cost Calculator</h2>
        <p className="estimator-subtitle">Real-time pricing based on your requirements</p>
      </div>

      <div className="estimator-container">
        {/* Input Controls */}
        <div className="estimator-inputs">
          <div className="input-group">
            <label htmlFor="duration-input">
              Duration (seconds)
              <span className="input-hint">{duration}s = {getDurationMinutes()} min</span>
            </label>
            <input
              id="duration-input"
              type="range"
              min="1"
              max="3600"
              value={duration}
              onChange={handleDurationChange}
              className="duration-slider"
              style={{
                background: `linear-gradient(to right, ${getTierColor(tier)} 0%, ${getTierColor(tier)} ${(duration / 3600) * 100}%, #e5e7eb ${(duration / 3600) * 100}%, #e5e7eb 100%)`,
              }}
            />
            <input
              type="number"
              min="1"
              max="3600"
              value={duration}
              onChange={handleDurationChange}
              className="duration-input-number"
            />
          </div>

          <div className="input-group">
            <label htmlFor="complexity-select">Video Complexity</label>
            <select
              id="complexity-select"
              value={complexity}
              onChange={handleComplexityChange}
              className="complexity-select"
            >
              <option value="simple">Simple (single dialogue, minimal transitions)</option>
              <option value="moderate">Moderate (multi-scene, standard transitions)</option>
              <option value="complex">Complex (VFX, multiple characters, advanced effects)</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="rush-select">Delivery Speed</label>
            <select
              id="rush-select"
              value={rushDelivery}
              onChange={handleRushDeliveryChange}
              className="rush-select"
            >
              <option value="standard">Standard Delivery (30-120 seconds)</option>
              <option value="express_6h">Express 6 Hours (+50%)</option>
              <option value="express_2h">Express 2 Hours (+150%)</option>
              <option value="on_demand">On-Demand (5 minutes) (+400%)</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="batch-input">
              Batch Order
              <span className="input-hint">{batchCount} video{batchCount !== 1 ? 's' : ''}</span>
            </label>
            <input
              id="batch-input"
              type="number"
              min="1"
              max="100"
              value={batchCount}
              onChange={handleBatchCountChange}
              className="batch-input"
            />
            <div className="batch-discount-info">
              {batchCount >= 25 && <span className="discount-badge">15% Discount Applied</span>}
              {batchCount >= 10 && batchCount < 25 && <span className="discount-badge">10% Discount Applied</span>}
              {batchCount >= 5 && batchCount < 10 && <span className="discount-badge">5% Discount Applied</span>}
              {batchCount < 5 && <span className="discount-badge">No Volume Discount</span>}
            </div>
          </div>
        </div>

        {/* Cost Breakdown Display */}
        <div className="cost-breakdown">
          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Calculating cost...</p>
            </div>
          )}

          {currentCost && !loading && (
            <>
              <div className="cost-header">
                <h3>Price Breakdown</h3>
                <div className="tier-badge" style={{ backgroundColor: getTierColor(tier) }}>
                  {tier.toUpperCase()} Tier
                </div>
              </div>

              <div className="cost-steps">
                {/* Base Cost */}
                <div className="cost-step">
                  <div className="step-label">
                    <span className="step-number">1</span>
                    <span className="step-title">Base Rate</span>
                    <span className="step-calc">
                      {getDurationMinutes()} min × ${currentCost.base_cost_per_min?.toFixed(2)}/min
                    </span>
                  </div>
                  <div className="step-amount">${currentCost.base_cost?.toFixed(3)}</div>
                </div>

                {/* Complexity Multiplier */}
                <div className="cost-step">
                  <div className="step-label">
                    <span className="step-number">2</span>
                    <span className="step-title">Complexity Adjustment</span>
                    <span className="step-calc">
                      ${currentCost.base_cost?.toFixed(3)} × {currentCost.multipliers?.complexity}x
                    </span>
                  </div>
                  <div className="step-amount">${currentCost.complexity_adjusted?.toFixed(3)}</div>
                </div>

                {/* Rush Delivery Multiplier */}
                {currentCost.multipliers?.rush_delivery > 1 && (
                  <div className="cost-step">
                    <div className="step-label">
                      <span className="step-number">3</span>
                      <span className="step-title">Rush Delivery</span>
                      <span className="step-calc">
                        × {currentCost.multipliers?.rush_delivery}x
                      </span>
                    </div>
                    <div className="step-amount">${currentCost.with_rush?.toFixed(3)}</div>
                  </div>
                )}

                {/* Batch Discount */}
                {currentCost.multipliers?.batch_discount !== '0%' && (
                  <div className="cost-step discount">
                    <div className="step-label">
                      <span className="step-number">4</span>
                      <span className="step-title">Volume Discount</span>
                      <span className="step-calc">
                        {currentCost.multipliers?.batch_discount}
                      </span>
                    </div>
                    <div className="step-amount negative">
                      -${(currentCost.with_rush - currentCost.with_batch)?.toFixed(3)}
                    </div>
                  </div>
                )}

                {/* Subtotal */}
                <div className="cost-step subtotal">
                  <div className="step-label">
                    <span className="step-title">Subtotal</span>
                  </div>
                  <div className="step-amount">${currentCost.subtotal?.replace('$', '')}</div>
                </div>

                {/* Tax */}
                <div className="cost-step tax">
                  <div className="step-label">
                    <span className="step-title">Sales Tax (8%)</span>
                  </div>
                  <div className="step-amount">${currentCost.tax?.replace('$', '')}</div>
                </div>
              </div>

              {/* Total Cost */}
              <div className="total-cost">
                <div className="total-label">Total Cost</div>
                <div className="total-amount" style={{ color: getTierColor(tier) }}>
                  {currentCost.total?.replace('$', '')}
                </div>
              </div>

              {/* Delivery Time */}
              <div className="delivery-info">
                <span className="delivery-icon">⏱️</span>
                <span className="delivery-text">
                  Estimated Delivery: {currentCost.estimated_delivery_sec}s
                </span>
              </div>
            </>
          )}

          {!currentCost && !loading && (
            <div className="empty-state">
              <p>Select parameters to calculate cost</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CostEstimator;
