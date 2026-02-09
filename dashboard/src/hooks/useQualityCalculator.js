import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Hook for interacting with Quality First API
 * Handles cost calculations, tier specs, and validations
 */
export const useQualityCalculator = (apiBaseUrl = 'http://localhost:8000') => {
  const [tierSpecs, setTierSpecs] = useState([]);
  const [costBreakdown, setCostBreakdown] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const debounceTimer = useRef(null);

  // Load tier specifications on mount
  useEffect(() => {
    loadTierSpecs();
  }, []);

  const loadTierSpecs = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/quality/tiers`);
      if (!response.ok) throw new Error('Failed to load tier specs');
      const data = await response.json();
      setTierSpecs(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error loading tier specs:', err);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  const calculateCost = useCallback(async (params) => {
    const {
      tier = 'good',
      duration_sec = 60,
      complexity = 'moderate',
      rush_delivery = 'standard',
      batch_count = 1,
      show_alternatives = false,
    } = params;

    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/quality/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tier,
          duration_sec,
          complexity,
          rush_delivery,
          batch_count,
          show_alternatives,
        }),
      });

      if (!response.ok) throw new Error('Failed to calculate cost');
      const data = await response.json();
      setCostBreakdown(data);
      setError(null);
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error calculating cost:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  const debouncedCalculateCost = useCallback((params) => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    debounceTimer.current = setTimeout(() => {
      calculateCost(params);
    }, 300); // 300ms debounce
  }, [calculateCost]);

  const updateCostEstimate = useCallback((params) => {
    debouncedCalculateCost(params);
  }, [debouncedCalculateCost]);

  const validateVideo = useCallback(async (jobId, tier, metadata) => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/quality/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          tier,
          video_metadata: metadata,
        }),
      });

      if (!response.ok) throw new Error('Failed to validate video');
      const data = await response.json();
      setError(null);
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error validating video:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  const getTierSpec = useCallback((tierName) => {
    return tierSpecs.find(t => t.tier === tierName);
  }, [tierSpecs]);

  const getAllAlternatives = useCallback(async (params) => {
    const {
      duration_sec = 60,
      complexity = 'moderate',
      batch_count = 1,
    } = params;

    try {
      setLoading(true);
      const goodCost = await calculateCost({ tier: 'good', duration_sec, complexity, batch_count });
      const highCost = await calculateCost({ tier: 'high', duration_sec, complexity, batch_count });
      const ultraCost = await calculateCost({ tier: 'ultra', duration_sec, complexity, batch_count });
      
      return {
        good: goodCost,
        high: highCost,
        ultra: ultraCost,
      };
    } catch (err) {
      setError('Failed to get tier alternatives');
      return null;
    } finally {
      setLoading(false);
    }
  }, [calculateCost]);

  return {
    tierSpecs,
    costBreakdown,
    loading,
    error,
    calculateCost,
    updateCostEstimate,
    validateVideo,
    getTierSpec,
    getAllAlternatives,
    loadTierSpecs,
  };
};
