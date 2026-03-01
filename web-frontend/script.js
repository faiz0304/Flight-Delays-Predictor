// Flight Delay Prediction Platform - JavaScript Frontend
// API integration and UI functionality

class FlightDelayPredictor {
    constructor() {
        this.apiBaseUrl = 'https://faiz720-flight-delays-api.hf.space'; // Rendered Hugging Face Backend URL
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setTodayAsDefaultDate();
    }

    setupEventListeners() {
        const form = document.getElementById('predictionForm');
        form.addEventListener('submit', (e) => this.handlePrediction(e));
    }

    setTodayAsDefaultDate() {
        const today = new Date().toISOString().split('T')[0];
        document.getElementById('flightDate').value = today;
    }

    handlePrediction(e) {
        e.preventDefault();

        // Show loading
        this.showLoading();

        // Get form data
        const formData = this.getFormData();

        // Call API
        this.callPredictionAPI(formData)
            .then(response => {
                this.displayResults(response);
            })
            .catch(error => {
                this.hideLoading();
                this.showError('Error: ' + error.message);
            });
    }

    getFormData() {
        const airline = document.getElementById('airline').value;
        const origin = document.getElementById('origin').value;
        const destination = document.getElementById('destination').value;
        const flightDate = new Date(document.getElementById('flightDate').value);

        const month = flightDate.getMonth() + 1; // JavaScript months are 0-indexed
        const dayOfWeek = flightDate.getDay() === 0 ? 7 : flightDate.getDay(); // Convert to 1-7 (1=Monday, 7=Sunday)

        return {
            airline: airline,
            origin: origin,
            destination: destination,
            month: month,
            day_of_week: dayOfWeek,
            distance: parseFloat(document.getElementById('distance').value),
            scheduled_departure_hour: parseInt(document.getElementById('departureHour').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            weather_condition: document.getElementById('weatherCondition').value
        };
    }

    async callPredictionAPI(flightData) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(flightData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Prediction API error:', error);
            throw error;
        }
    }

    displayResults(prediction) {
        this.hideLoading();

        // Update prediction results
        const delayProbability = (prediction.delay_probability * 100).toFixed(1);
        const predictionResult = prediction.prediction;
        const delayMinutes = prediction.delay_minutes.toFixed(1);

        document.getElementById('delayProbability').textContent = `${delayProbability}%`;
        document.getElementById('predictionResult').textContent = predictionResult;
        document.getElementById('delayMinutes').textContent = `${delayMinutes} minutes`;

        // Update risk level and styling
        this.updateRiskLevel(prediction.details.risk_level, delayProbability);

        // Update prediction result styling
        const predictionResultElement = document.getElementById('predictionResult');
        predictionResultElement.className = predictionResult === 'Delayed' ? 'delayed' : 'on-time';

        // Show results section
        document.getElementById('resultsSection').style.display = 'block';

        // Display flight information
        this.displayFlightInfo();

        // Create risk factor chart
        this.createRiskChart(prediction.details.flight_info, delayProbability);
    }

    updateRiskLevel(riskLevel, probability) {
        const riskLevelElement = document.getElementById('riskLevel');
        riskLevelElement.textContent = `${riskLevel} Risk`;
        riskLevelElement.className = `risk-level risk-${riskLevel.toLowerCase()}`;

        // Update probability card background based on risk
        const probabilityCard = document.getElementById('delayProbabilityCard');
        if (probability > 70) {
            probabilityCard.style.background = 'linear-gradient(135deg, #f56565 0%, #fc8181 100%)';
            probabilityCard.style.color = 'white';
        } else if (probability > 40) {
            probabilityCard.style.background = 'linear-gradient(135deg, #ecc94b 0%, #f6e05e 100%)';
            probabilityCard.style.color = 'white';
        } else {
            probabilityCard.style.background = 'linear-gradient(135deg, #48bb78 0%, #68d391 100%)';
            probabilityCard.style.color = 'white';
        }
    }

    displayFlightInfo() {
        const flightInfoElement = document.getElementById('flightInfo');
        const flightInfo = this.getFormData();

        const airlineNames = {
            'AA': 'American Airlines',
            'UA': 'United Airlines',
            'DL': 'Delta Air Lines',
            'SW': 'Southwest Airlines',
            'B6': 'JetBlue Airways',
            'AS': 'Alaska Airlines',
            'NK': 'Spirit Airlines',
            'F9': 'Frontier Airlines',
            'HA': 'Hawaiian Airlines'
        };

        flightInfoElement.innerHTML = `
            <p><strong>Airline:</strong> ${airlineNames[flightInfo.airline]} (${flightInfo.airline})</p>
            <p><strong>Route:</strong> ${flightInfo.origin} → ${flightInfo.destination}</p>
            <p><strong>Date:</strong> ${new Date(document.getElementById('flightDate').value).toLocaleDateString()}</p>
            <p><strong>Distance:</strong> ${flightInfo.distance} miles</p>
            <p><strong>Departure:</strong> ${flightInfo.scheduled_departure_hour}:00</p>
            <p><strong>Weather:</strong> ${flightInfo.weather_condition}</p>
            <p><strong>Temperature:</strong> ${flightInfo.temperature}°C</p>
            <p><strong>Humidity:</strong> ${flightInfo.humidity}%</p>
        `;
    }

    createRiskChart(flightInfo, delayProbability) {
        const ctx = document.getElementById('riskChart').getContext('2d');

        // Calculate risk factors based on input
        const factors = [
            { label: 'Weather', value: this.getWeatherRisk(flightInfo.weather_condition) },
            { label: 'Time of Day', value: this.getTimeRisk(flightInfo.scheduled_departure_hour) },
            { label: 'Day of Week', value: this.getDayRisk(flightInfo.day_of_week) },
            { label: 'Month', value: this.getMonthRisk(flightInfo.month) },
            { label: 'Distance', value: this.getDistanceRisk(flightInfo.distance) }
        ];

        // Destroy existing chart if exists
        if (window.riskChart) {
            try {
                window.riskChart.destroy();
            } catch (error) {
                console.log('Chart already destroyed or not initialized');
            }
        }

        // Wait for any pending animations to complete before creating new chart
        setTimeout(() => {
            try {
                window.riskChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: factors.map(f => f.label),
                        datasets: [{
                            label: 'Risk Impact',
                            data: factors.map(f => f.value),
                            backgroundColor: [
                                'rgba(245, 101, 101, 0.8)',  // Weather - Red
                                'rgba(236, 201, 75, 0.8)',   // Time - Yellow
                                'rgba(72, 187, 120, 0.8)',   // Day - Green
                                'rgba(102, 126, 234, 0.8)', // Month - Blue
                                'rgba(156, 136, 255, 0.8)'  // Distance - Purple
                            ],
                            borderColor: [
                                'rgba(245, 101, 101, 1)',
                                'rgba(236, 201, 75, 1)',
                                'rgba(72, 187, 120, 1)',
                                'rgba(102, 126, 234, 1)',
                                'rgba(156, 136, 255, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: {
                            duration: 1000,
                            easing: 'easeOutQuart'
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Risk Factor Impact',
                                font: {
                                    size: 16
                                }
                            }
                        },
                        scales: {
                            x: {
                                beginAtZero: true,
                                max: 1,
                                title: {
                                    display: true,
                                    text: 'Risk Level (0-1)'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Risk Factors'
                                }
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Error creating chart:', error);
            }
        }, 100); // Small delay to ensure canvas is ready
    }

    getWeatherRisk(weather) {
        const weatherRisk = {
            'Clear': 0.1,
            'Cloudy': 0.2,
            'Rain': 0.4,
            'Snow': 0.6,
            'Storm': 0.8
        };
        return weatherRisk[weather] || 0.1;
    }

    getTimeRisk(hour) {
        // Peak hours: 6-8 AM and 4-6 PM
        if ((hour >= 6 && hour <= 8) || (hour >= 16 && hour <= 18)) {
            return 0.7;
        } else if (hour >= 5 && hour <= 20) {
            return 0.4; // Business hours
        } else {
            return 0.2; // Night hours
        }
    }

    getDayRisk(day) {
        // Weekend (Friday-Sunday) has higher risk
        if (day === 5 || day === 6 || day === 7) {
            return 0.6;
        } else {
            return 0.3;
        }
    }

    getMonthRisk(month) {
        // Winter months (Dec, Jan, Feb) and summer travel (Jun, Jul, Aug) have higher risk
        if (month === 12 || month === 1 || month === 2) {
            return 0.7;
        } else if (month >= 6 && month <= 8) {
            return 0.6;
        } else {
            return 0.3;
        }
    }

    getDistanceRisk(distance) {
        // Longer distances have slightly higher risk
        if (distance > 1500) {
            return 0.6;
        } else if (distance > 800) {
            return 0.4;
        } else {
            return 0.2;
        }
    }

    showLoading() {
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
        document.querySelector('.predict-btn').disabled = true;
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
        document.querySelector('.predict-btn').disabled = false;
    }

    showError(message) {
        // Create error message element
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            background: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #feb2b2;
        `;
        errorDiv.textContent = message;

        // Remove existing error messages
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        // Insert error message before the form
        const form = document.getElementById('predictionForm');
        form.parentNode.insertBefore(errorDiv, form);

        // Scroll to error
        errorDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FlightDelayPredictor();
});

// Add a test function for API connectivity
async function testApiConnection() {
    try {
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        if (data.status === 'healthy' && data.models_loaded) {
            console.log('API connection successful');
        } else {
            console.warn('API may not be ready:', data);
        }
    } catch (error) {
        console.error('API connection failed:', error);
        console.log('Note: Backend API must be running on http://localhost:8000');
    }
}

// Test API connection on load
testApiConnection();