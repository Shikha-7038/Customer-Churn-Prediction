"""
FastAPI Service for Churn Prediction
Provides REST endpoints for scoring and explanation
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from src
from src.features import add_features
from src.pipeline import create_preprocessor

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict which customers are likely to churn and get actionable insights",
    version="1.0.0",
    docs_url="/swagger",  # Swagger UI at /swagger
    redoc_url="/redoc"     # ReDoc at /redoc
)

# Global variables for models
model = None
preprocessor = None

# Model paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")

def load_models():
    """Load models at startup"""
    global model, preprocessor
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"[OK] Model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] Model not found at {MODEL_PATH}")
            
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"[OK] Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            print(f"[WARN] Preprocessor not found at {PREPROCESSOR_PATH}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

# Load models on startup
load_models()

# Define input schema
class CustomerData(BaseModel):
    customer_id: Optional[str] = None
    billing_amount: float = Field(..., ge=0, le=500, description="Monthly billing amount")
    last_payment_days_ago: float = Field(..., ge=0, le=90, description="Days since last payment")
    plan_tier: str = Field(..., description="Plan type: Basic, Standard, or Premium")
    tenure_months: float = Field(..., ge=0, le=120, description="Customer tenure in months")
    monthly_usage_hours: float = Field(..., ge=0, le=500, description="Monthly usage in hours")
    active_days: float = Field(..., ge=0, le=30, description="Number of active days in cycle")
    login_count: float = Field(..., ge=0, description="Number of logins")
    avg_session_min: float = Field(..., ge=0, description="Average session length in minutes")
    device_count: float = Field(..., ge=0, description="Number of devices used")
    add_on_count: float = Field(..., ge=0, description="Number of add-ons purchased")
    support_tickets: float = Field(..., ge=0, description="Number of support tickets")
    sla_breaches: float = Field(..., ge=0, description="Number of SLA breaches")
    promotions_redeemed: float = Field(..., ge=0, description="Number of promotions redeemed")
    email_opens: float = Field(..., ge=0, description="Number of email opens")
    email_clicks: float = Field(..., ge=0, description="Number of email clicks")
    last_campaign_days_ago: float = Field(..., ge=0, description="Days since last campaign")
    nps_score: float = Field(..., ge=0, le=10, description="NPS score (0-10)")
    region: str = Field(..., description="Region: North, South, East, or West")
    is_autopay: bool = Field(..., description="Whether autopay is enabled")
    is_discounted: bool = Field(..., description="Whether customer has discount")
    has_family_bundle: bool = Field(..., description="Whether customer has family bundle")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "billing_amount": 49.99,
                "last_payment_days_ago": 5,
                "plan_tier": "Standard",
                "tenure_months": 12,
                "monthly_usage_hours": 45,
                "active_days": 22,
                "login_count": 30,
                "avg_session_min": 25,
                "device_count": 2,
                "add_on_count": 1,
                "support_tickets": 0,
                "sla_breaches": 0,
                "promotions_redeemed": 1,
                "email_opens": 5,
                "email_clicks": 2,
                "last_campaign_days_ago": 10,
                "nps_score": 8,
                "region": "North",
                "is_autopay": True,
                "is_discounted": False,
                "has_family_bundle": False
            }
        }

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    risk_segment: str
    recommended_action: str

class ExplanationResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    top_factors: List[Dict[str, float]]
    recommended_action: str

# Helper functions
def get_risk_segment(probability: float) -> str:
    """Determine risk segment based on probability"""
    if probability >= 0.5:
        return "High Risk"
    elif probability >= 0.25:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_recommended_action(probability: float, customer_df: pd.DataFrame) -> str:
    """Get recommended action based on probability and customer features"""
    if probability >= 0.5:
        if customer_df['support_tickets'].iloc[0] > 2:
            return "Immediate support escalation and retention offer"
        elif customer_df['monthly_usage_hours'].iloc[0] < 20:
            return "Engagement campaign with usage tips and discount"
        elif not customer_df['is_autopay'].iloc[0]:
            return "Payment reminder with autopay incentive"
        else:
            return "Priority retention outreach with personalized offer"
    elif probability >= 0.25:
        if customer_df['nps_score'].iloc[0] < 6:
            return "Satisfaction survey and check-in call"
        else:
            return "Targeted email campaign with product updates"
    else:
        if customer_df['plan_tier'].iloc[0] == 'Basic' and customer_df['monthly_usage_hours'].iloc[0] > 50:
            return "Upsell opportunity - Premium plan recommendation"
        else:
            return "Maintain regular engagement and loyalty rewards"

def get_top_factors(customer_df: pd.DataFrame, probability: float) -> List[Dict[str, float]]:
    """Get top factors contributing to churn risk"""
    factors = []
    
    engagement_rate = customer_df['active_days'].iloc[0] / 30
    if engagement_rate < 0.5:
        factors.append({"factor": "Low Engagement", "impact": min(0.3, probability)})
    
    if customer_df['support_tickets'].iloc[0] > 2:
        factors.append({"factor": "High Support Tickets", "impact": min(0.25, probability)})
    
    if customer_df['last_payment_days_ago'].iloc[0] > 30:
        factors.append({"factor": "Late Payment", "impact": min(0.2, probability)})
    
    if customer_df['tenure_months'].iloc[0] < 3:
        factors.append({"factor": "New Customer (High Risk Period)", "impact": min(0.15, probability)})
    
    if customer_df['nps_score'].iloc[0] < 5:
        factors.append({"factor": "Low Satisfaction Score", "impact": min(0.1, probability)})
    
    if customer_df['plan_tier'].iloc[0] == 'Premium' and customer_df['monthly_usage_hours'].iloc[0] < 20:
        factors.append({"factor": "Underutilizing Premium Plan", "impact": min(0.12, probability)})
    
    return factors[:5]

# ============================================
# INTERACTIVE DOCS ENDPOINT - With Testing Interface
# ============================================

@app.get("/docs", response_class=HTMLResponse)
async def interactive_docs():
    """Interactive documentation page where you can actually test the API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Prediction API - Interactive Docs</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #f5f5f5;
                min-height: 100vh;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .endpoint-card {
                background: white;
                border-radius: 12px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .endpoint-header {
                background: #f8f9fa;
                padding: 15px 20px;
                border-bottom: 2px solid #e9ecef;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .endpoint-header:hover {
                background: #e9ecef;
            }
            
            .method {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 0.85em;
                margin-right: 15px;
            }
            
            .method-get {
                background: #10b981;
                color: white;
            }
            
            .method-post {
                background: #3b82f6;
                color: white;
            }
            
            .url {
                font-family: 'Courier New', monospace;
                font-size: 1.1em;
                color: #333;
            }
            
            .endpoint-body {
                padding: 20px;
                display: none;
            }
            
            .endpoint-body.active {
                display: block;
            }
            
            .input-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
            }
            
            textarea, select, input {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            
            textarea {
                min-height: 300px;
                resize: vertical;
            }
            
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                margin-right: 10px;
            }
            
            button:hover {
                background: #5a67d8;
            }
            
            .response-area {
                margin-top: 20px;
                border-top: 1px solid #e9ecef;
                padding-top: 20px;
            }
            
            .response {
                background: #1e1e2e;
                color: #e5e7eb;
                padding: 15px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                white-space: pre-wrap;
            }
            
            .status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                margin-bottom: 10px;
            }
            
            .status-success {
                background: #10b981;
                color: white;
            }
            
            .status-error {
                background: #ef4444;
                color: white;
            }
            
            .preset-buttons {
                margin-bottom: 15px;
            }
            
            .preset-btn {
                background: #e5e7eb;
                color: #333;
                padding: 6px 12px;
                font-size: 0.85em;
                margin-right: 10px;
            }
            
            .preset-btn:hover {
                background: #d1d5db;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-left: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .grid-3 {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .stat-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Customer Churn Prediction API</h1>
            <p>Interactive documentation - Test the API directly from your browser!</p>
        </div>
        
        <div class="container">
            <!-- Status Cards -->
            <div class="grid-3" id="statusCards">
                <div class="stat-card">
                    <h3>🟢 API Status</h3>
                    <p id="apiStatus">Checking...</p>
                </div>
                <div class="stat-card">
                    <h3>🤖 Model Status</h3>
                    <p id="modelStatus">Checking...</p>
                </div>
                <div class="stat-card">
                    <h3>📊 Model Type</h3>
                    <p id="modelType">XGBoost</p>
                </div>
            </div>
            
            <!-- Endpoint 1: Health Check -->
            <div class="endpoint-card">
                <div class="endpoint-header" onclick="toggleEndpoint(this)">
                    <div>
                        <span class="method method-get">GET</span>
                        <span class="url">/health</span>
                    </div>
                    <span>▼</span>
                </div>
                <div class="endpoint-body">
                    <p style="margin-bottom: 15px;">Check if the API and model are healthy and ready.</p>
                    <button onclick="testHealth()">▶ Test Endpoint</button>
                    <div id="healthResponse" class="response-area"></div>
                </div>
            </div>
            
            <!-- Endpoint 2: Score (Single Prediction) -->
            <div class="endpoint-card">
                <div class="endpoint-header" onclick="toggleEndpoint(this)">
                    <div>
                        <span class="method method-post">POST</span>
                        <span class="url">/score</span>
                    </div>
                    <span>▼</span>
                </div>
                <div class="endpoint-body">
                    <p style="margin-bottom: 15px;">Predict churn probability for a single customer.</p>
                    
                    <div class="preset-buttons">
                        <span style="margin-right: 10px;">📋 Presets:</span>
                        <button class="preset-btn" onclick="loadPreset('low')">Low Risk Customer</button>
                        <button class="preset-btn" onclick="loadPreset('medium')">Medium Risk Customer</button>
                        <button class="preset-btn" onclick="loadPreset('high')">High Risk Customer</button>
                        <button class="preset-btn" onclick="loadPreset('example')">Example Customer</button>
                    </div>
                    
                    <div class="input-group">
                        <label>Request Body (JSON):</label>
                        <textarea id="scoreInput" placeholder='{"customer_id": "CUST_001", ...}'></textarea>
                    </div>
                    
                    <button onclick="testScore()">🚀 Send Request</button>
                    <button onclick="clearResponse('scoreResponse')">🗑 Clear</button>
                    
                    <div id="scoreResponse" class="response-area"></div>
                </div>
            </div>
            
            <!-- Endpoint 3: Explain -->
            <div class="endpoint-card">
                <div class="endpoint-header" onclick="toggleEndpoint(this)">
                    <div>
                        <span class="method method-post">POST</span>
                        <span class="url">/explain</span>
                    </div>
                    <span>▼</span>
                </div>
                <div class="endpoint-body">
                    <p style="margin-bottom: 15px;">Get churn prediction with explanation of key risk factors.</p>
                    
                    <div class="preset-buttons">
                        <span style="margin-right: 10px;">📋 Presets:</span>
                        <button class="preset-btn" onclick="loadPresetForExplain('low')">Low Risk Customer</button>
                        <button class="preset-btn" onclick="loadPresetForExplain('high')">High Risk Customer</button>
                    </div>
                    
                    <div class="input-group">
                        <label>Request Body (JSON):</label>
                        <textarea id="explainInput" placeholder='{"customer_id": "CUST_001", ...}'></textarea>
                    </div>
                    
                    <button onclick="testExplain()">💡 Get Explanation</button>
                    <button onclick="clearResponse('explainResponse')">🗑 Clear</button>
                    
                    <div id="explainResponse" class="response-area"></div>
                </div>
            </div>
            
            <!-- Endpoint 4: Batch Score -->
            <div class="endpoint-card">
                <div class="endpoint-header" onclick="toggleEndpoint(this)">
                    <div>
                        <span class="method method-post">POST</span>
                        <span class="url">/batch_score</span>
                    </div>
                    <span>▼</span>
                </div>
                <div class="endpoint-body">
                    <p style="margin-bottom: 15px;">Predict churn for multiple customers at once.</p>
                    
                    <div class="input-group">
                        <label>Request Body (JSON):</label>
                        <textarea id="batchInput" placeholder='{"customers": [...]}'></textarea>
                    </div>
                    
                    <button onclick="testBatch()">📊 Send Batch Request</button>
                    <button onclick="clearResponse('batchResponse')">🗑 Clear</button>
                    
                    <div id="batchResponse" class="response-area"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Customer Churn Prediction API v1.0.0 | <a href="/swagger">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
        </div>
        
        <script>
            // Sample data presets
            const presets = {
                low: {
                    "customer_id": "LOW_001",
                    "billing_amount": 29.99,
                    "last_payment_days_ago": 2,
                    "plan_tier": "Basic",
                    "tenure_months": 36,
                    "monthly_usage_hours": 120,
                    "active_days": 28,
                    "login_count": 80,
                    "avg_session_min": 35,
                    "device_count": 3,
                    "add_on_count": 2,
                    "support_tickets": 0,
                    "sla_breaches": 0,
                    "promotions_redeemed": 2,
                    "email_opens": 10,
                    "email_clicks": 5,
                    "last_campaign_days_ago": 3,
                    "nps_score": 9,
                    "region": "North",
                    "is_autopay": true,
                    "is_discounted": false,
                    "has_family_bundle": true
                },
                medium: {
                    "customer_id": "MED_001",
                    "billing_amount": 49.99,
                    "last_payment_days_ago": 15,
                    "plan_tier": "Standard",
                    "tenure_months": 12,
                    "monthly_usage_hours": 45,
                    "active_days": 18,
                    "login_count": 25,
                    "avg_session_min": 20,
                    "device_count": 2,
                    "add_on_count": 1,
                    "support_tickets": 1,
                    "sla_breaches": 0,
                    "promotions_redeemed": 1,
                    "email_opens": 3,
                    "email_clicks": 1,
                    "last_campaign_days_ago": 20,
                    "nps_score": 6,
                    "region": "East",
                    "is_autopay": true,
                    "is_discounted": true,
                    "has_family_bundle": false
                },
                high: {
                    "customer_id": "HIGH_001",
                    "billing_amount": 99.99,
                    "last_payment_days_ago": 45,
                    "plan_tier": "Premium",
                    "tenure_months": 2,
                    "monthly_usage_hours": 8,
                    "active_days": 3,
                    "login_count": 2,
                    "avg_session_min": 5,
                    "device_count": 1,
                    "add_on_count": 0,
                    "support_tickets": 5,
                    "sla_breaches": 3,
                    "promotions_redeemed": 0,
                    "email_opens": 0,
                    "email_clicks": 0,
                    "last_campaign_days_ago": 60,
                    "nps_score": 2,
                    "region": "South",
                    "is_autopay": false,
                    "is_discounted": false,
                    "has_family_bundle": false
                },
                example: {
                    "customer_id": "EXAMPLE_001",
                    "billing_amount": 49.99,
                    "last_payment_days_ago": 5,
                    "plan_tier": "Standard",
                    "tenure_months": 12,
                    "monthly_usage_hours": 45,
                    "active_days": 22,
                    "login_count": 30,
                    "avg_session_min": 25,
                    "device_count": 2,
                    "add_on_count": 1,
                    "support_tickets": 0,
                    "sla_breaches": 0,
                    "promotions_redeemed": 1,
                    "email_opens": 5,
                    "email_clicks": 2,
                    "last_campaign_days_ago": 10,
                    "nps_score": 8,
                    "region": "North",
                    "is_autopay": true,
                    "is_discounted": false,
                    "has_family_bundle": false
                }
            };
            
            // Load health status
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('apiStatus').innerHTML = '🟢 Online';
                    document.getElementById('modelStatus').innerHTML = data.model_loaded ? '✅ Loaded' : '❌ Not Loaded';
                } catch (error) {
                    document.getElementById('apiStatus').innerHTML = '🔴 Offline';
                    document.getElementById('modelStatus').innerHTML = '❌ Error';
                }
            }
            
            // Toggle endpoint visibility
            function toggleEndpoint(header) {
                const body = header.nextElementSibling;
                body.classList.toggle('active');
                const arrow = header.querySelector('span:last-child');
                arrow.textContent = body.classList.contains('active') ? '▲' : '▼';
            }
            
            // Load preset for score endpoint
            function loadPreset(type) {
                const preset = presets[type];
                document.getElementById('scoreInput').value = JSON.stringify(preset, null, 2);
            }
            
            // Load preset for explain endpoint
            function loadPresetForExplain(type) {
                const preset = presets[type];
                document.getElementById('explainInput').value = JSON.stringify(preset, null, 2);
            }
            
            // Clear response
            function clearResponse(elementId) {
                document.getElementById(elementId).innerHTML = '';
            }
            
            // Test health endpoint
            async function testHealth() {
                const responseDiv = document.getElementById('healthResponse');
                responseDiv.innerHTML = '<div class="loading"></div> Testing...';
                
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    responseDiv.innerHTML = `
                        <div class="status status-success">✓ Status: ${response.status}</div>
                        <div class="response">${JSON.stringify(data, null, 2)}</div>
                    `;
                } catch (error) {
                    responseDiv.innerHTML = `
                        <div class="status status-error">✗ Error</div>
                        <div class="response">${error.message}</div>
                    `;
                }
            }
            
            // Test score endpoint
            async function testScore() {
                const input = document.getElementById('scoreInput').value;
                const responseDiv = document.getElementById('scoreResponse');
                
                if (!input) {
                    responseDiv.innerHTML = '<div class="status status-error">Please enter request data</div>';
                    return;
                }
                
                responseDiv.innerHTML = '<div class="loading"></div> Sending request...';
                
                try {
                    const response = await fetch('/score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: input
                    });
                    
                    const data = await response.json();
                    const statusClass = response.ok ? 'status-success' : 'status-error';
                    
                    // Format probability as percentage
                    if (data.churn_probability) {
                        data.churn_probability_percent = (data.churn_probability * 100).toFixed(1) + '%';
                    }
                    
                    responseDiv.innerHTML = `
                        <div class="status ${statusClass}">${response.ok ? '✓ Success' : '✗ Error'} - Status: ${response.status}</div>
                        <div class="response">${JSON.stringify(data, null, 2)}</div>
                    `;
                } catch (error) {
                    responseDiv.innerHTML = `
                        <div class="status status-error">✗ Request Failed</div>
                        <div class="response">${error.message}</div>
                    `;
                }
            }
            
            // Test explain endpoint
            async function testExplain() {
                const input = document.getElementById('explainInput').value;
                const responseDiv = document.getElementById('explainResponse');
                
                if (!input) {
                    responseDiv.innerHTML = '<div class="status status-error">Please enter request data</div>';
                    return;
                }
                
                responseDiv.innerHTML = '<div class="loading"></div> Getting explanation...';
                
                try {
                    const response = await fetch('/explain', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: input
                    });
                    
                    const data = await response.json();
                    const statusClass = response.ok ? 'status-success' : 'status-error';
                    
                    responseDiv.innerHTML = `
                        <div class="status ${statusClass}">${response.ok ? '✓ Success' : '✗ Error'} - Status: ${response.status}</div>
                        <div class="response">${JSON.stringify(data, null, 2)}</div>
                    `;
                } catch (error) {
                    responseDiv.innerHTML = `
                        <div class="status status-error">✗ Request Failed</div>
                        <div class="response">${error.message}</div>
                    `;
                }
            }
            
            // Test batch endpoint
            async function testBatch() {
                const input = document.getElementById('batchInput').value;
                const responseDiv = document.getElementById('batchResponse');
                
                if (!input) {
                    responseDiv.innerHTML = '<div class="status status-error">Please enter request data</div>';
                    return;
                }
                
                responseDiv.innerHTML = '<div class="loading"></div> Processing batch...';
                
                try {
                    const response = await fetch('/batch_score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: input
                    });
                    
                    const data = await response.json();
                    const statusClass = response.ok ? 'status-success' : 'status-error';
                    
                    responseDiv.innerHTML = `
                        <div class="status ${statusClass}">${response.ok ? '✓ Success' : '✗ Error'} - Status: ${response.status}</div>
                        <div class="response">${JSON.stringify(data, null, 2)}</div>
                    `;
                } catch (error) {
                    responseDiv.innerHTML = `
                        <div class="status status-error">✗ Request Failed</div>
                        <div class="response">${error.message}</div>
                    `;
                }
            }
            
            // Load batch example
            const batchExample = {
                "customers": [presets.low, presets.medium, presets.high]
            };
            document.getElementById('batchInput').value = JSON.stringify(batchExample, null, 2);
            
            // Initialize
            checkHealth();
            
            // Open first endpoint by default
            setTimeout(() => {
                const firstEndpoint = document.querySelector('.endpoint-body');
                if (firstEndpoint) {
                    firstEndpoint.classList.add('active');
                }
            }, 500);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ============================================
# MAIN API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "swagger_ui": "/swagger",
        "redoc": "/redoc",
        "endpoints": {
            "/score": "POST - Predict churn for a single customer",
            "/batch_score": "POST - Predict churn for multiple customers",
            "/explain": "POST - Get explanation for a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive documentation with testing"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/score", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn probability for a single customer"""
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first by running python src/train_model.py")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.model_dump()
        customer_id = customer_dict.pop('customer_id')
        customer_df = pd.DataFrame([customer_dict])
        
        # Add engineered features
        customer_df = add_features(customer_df)
        
        # Get features for preprocessing
        _, num_feat, cat_feat = create_preprocessor()
        features = num_feat + cat_feat
        
        # Ensure all features exist
        available_features = [f for f in features if f in customer_df.columns]
        
        X = customer_df[available_features]
        
        # Transform and predict
        X_transformed = preprocessor.transform(X)
        
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X_transformed)[0, 1])
        else:
            # Fallback for models without predict_proba
            probability = float(model.predict(X_transformed)[0])
        
        # Get risk segment and action
        risk_segment = get_risk_segment(probability)
        recommended_action = get_recommended_action(probability, customer_df)
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(probability, 4),
            risk_segment=risk_segment,
            recommended_action=recommended_action
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@app.post("/batch_score")
async def batch_predict(batch: BatchCustomerData):
    """Predict churn for multiple customers"""
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        _, num_feat, cat_feat = create_preprocessor()
        features = num_feat + cat_feat
        
        for customer in batch.customers:
            customer_dict = customer.model_dump()
            customer_id = customer_dict.pop('customer_id')
            customer_df = pd.DataFrame([customer_dict])
            customer_df = add_features(customer_df)
            
            available_features = [f for f in features if f in customer_df.columns]
            X = customer_df[available_features]
            
            X_transformed = preprocessor.transform(X)
            
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(X_transformed)[0, 1])
            else:
                probability = float(model.predict(X_transformed)[0])
                
            risk_segment = get_risk_segment(probability)
            
            results.append({
                "customer_id": customer_id,
                "churn_probability": round(probability, 4),
                "risk_segment": risk_segment
            })
        
        return {"predictions": results, "total": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing batch: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(customer: CustomerData):
    """Get explanation for churn prediction"""
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.model_dump()
        customer_id = customer_dict.pop('customer_id')
        customer_df = pd.DataFrame([customer_dict])
        
        # Add engineered features
        customer_df = add_features(customer_df)
        
        # Get features for preprocessing
        _, num_feat, cat_feat = create_preprocessor()
        features = num_feat + cat_feat
        available_features = [f for f in features if f in customer_df.columns]
        X = customer_df[available_features]
        
        # Transform and predict
        X_transformed = preprocessor.transform(X)
        
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X_transformed)[0, 1])
        else:
            probability = float(model.predict(X_transformed)[0])
        
        # Get top factors
        top_factors = get_top_factors(customer_df, probability)
        recommended_action = get_recommended_action(probability, customer_df)
        
        return ExplanationResponse(
            customer_id=customer_id,
            churn_probability=round(probability, 4),
            top_factors=top_factors,
            recommended_action=recommended_action
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating explanation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)