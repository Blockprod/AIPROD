import React, { useState } from 'react';
import './App.css';
import { useQualityCalculator } from './hooks/useQualityCalculator';
import QualityTierSelector from './components/QualityTierSelector';
import CostEstimator from './components/CostEstimator';
import TierComparisonTable from './components/TierComparisonTable';
import QualityGuarantee from './components/QualityGuarantee';
import AlternativesDisplay from './components/AlternativesDisplay';

/**
 * AIPROD Dashboard - Quality First Framework
 * Integrates all quality and pricing components
 */
function App() {
  const [selectedTier, setSelectedTier] = useState('good');
  const [duration, setDuration] = useState(60);
  const [complexity, setComplexity] = useState('moderate');
  const [rushDelivery, setRushDelivery] = useState('standard');
  const [batchCount, setBatchCount] = useState(1);

  // Initialize quality calculator hook
  const {
    tierSpecs,
    costBreakdown,
    loading,
    error,
    updateCostEstimate,
    getAllAlternatives,
  } = useQualityCalculator('http://localhost:8000');

  const handleTierChange = (newTier) => {
    setSelectedTier(newTier);
  };

  const handleCostUpdate = (newCost) => {
    // Handle cost updates if needed
    console.log('Cost updated:', newCost);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>🎬 AIPROD Quality First</h1>
          <p className="header-subtitle">Professional video generation with guaranteed quality tiers</p>
        </div>
      </header>

      {/* Error Message */}
      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠️</span>
          <span className="error-message">{error}</span>
        </div>
      )}

      {/* Main Container */}
      <main className="app-main">
        {/* Section 1: Quality Tier Selection */}
        <section className="app-section" id="tier-section">
          <QualityTierSelector
            selectedTier={selectedTier}
            onTierChange={handleTierChange}
            tierSpecs={tierSpecs}
          />
        </section>

        {/* Section 2: Cost Estimation */}
        <section className="app-section" id="cost-section">
          <CostEstimator
            tier={selectedTier}
            onCostUpdate={handleCostUpdate}
            updateCostEstimate={updateCostEstimate}
            currentCost={costBreakdown}
            loading={loading}
          />
        </section>

        {/* Section 3: Quality Guarantee */}
        <section className="app-section" id="guarantee-section">
          <QualityGuarantee
            tier={selectedTier}
            certified={true}
            tierSpecs={tierSpecs}
          />
        </section>

        {/* Section 4: Tier Comparison */}
        <section className="app-section" id="comparison-section">
          <TierComparisonTable
            selectedTier={selectedTier}
            onTierSelect={handleTierChange}
            tierSpecs={tierSpecs}
          />
        </section>

        {/* Section 5: Alternatives Display */}
        <section className="app-section" id="alternatives-section">
          <AlternativesDisplay
            selectedTier={selectedTier}
            duration={duration}
            complexity={complexity}
            batchCount={batchCount}
            getAllAlternatives={getAllAlternatives}
            loading={loading}
          />
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>About Quality First</h4>
            <p>
              AIPROD Quality First ensures professional video generation with guaranteed quality
              standards. Every video is automatically validated and certified.
            </p>
          </div>

          <div className="footer-section">
            <h4>Support</h4>
            <ul>
              <li><a href="#quality-tiers">Quality Tiers</a></li>
              <li><a href="#pricing">Pricing</a></li>
              <li><a href="#docs">Documentation</a></li>
              <li><a href="#contact">Contact Us</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h4>Quality Standards</h4>
            <ul>
              <li>✓ GOOD: 1080p Professional</li>
              <li>✓ HIGH: 4K Broadcast</li>
              <li>✓ ULTRA: 4K@60fps HDR Cinema</li>
            </ul>
          </div>

          <div className="footer-section">
            <h4>Service Info</h4>
            <ul>
              <li>Status: ✓ Operational</li>
              <li>Uptime: 99.99%</li>
              <li>Support: 24/7</li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <p>&copy; 2026 AIPROD. All rights reserved. Quality First. Innovation Always.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
