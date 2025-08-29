#!/usr/bin/env python3
"""
AI Hospital Triage System - Python Backend
Fuzzy Logic Implementation for Medical Triage with Chatbot Integration
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sqlite3
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Patient:
    """Patient data structure"""
    id: int
    name: str
    age: int
    gender: str
    symptoms: Dict[str, float]
    additional_symptoms: str
    arrival_time: datetime
    priority: int
    priority_label: str
    urgency_score: float
    estimated_wait_time: int

    def to_dict(self):
        """Convert patient to dictionary for JSON serialization"""
        result = asdict(self)
        result['arrival_time'] = self.arrival_time.isoformat()
        return result

class FuzzyMembershipFunctions:
    """Fuzzy logic membership functions"""
    
    @staticmethod
    def triangular_membership(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function"""
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (d - x) / (d - c)

class FuzzyTriageSystem:
    """Main fuzzy logic triage system"""
    
    def __init__(self):
        self.patients = []
        self.patient_id_counter = 1
        self.membership = FuzzyMembershipFunctions()
        
        # Triage priority thresholds
        self.priority_thresholds = {
            1: 85,  # Critical
            2: 65,  # High
            3: 45,  # Moderate
            4: 25,  # Low
            5: 0    # Minimal
        }
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for patient records"""
        try:
            self.conn = sqlite3.connect('triage_system.db', check_same_thread=False)
            cursor = self.conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    additional_symptoms TEXT,
                    arrival_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    priority INTEGER NOT NULL,
                    priority_label TEXT NOT NULL,
                    urgency_score REAL NOT NULL,
                    estimated_wait_time INTEGER NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    session_id TEXT
                )
            ''')
            
            self.conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def pain_fuzzification(self, pain: float) -> Dict[str, float]:
        """Fuzzify pain level"""
        return {
            'none': self.membership.trapezoidal_membership(pain, 0, 0, 1, 2),
            'mild': self.membership.triangular_membership(pain, 0, 2, 4),
            'moderate': self.membership.triangular_membership(pain, 2, 5, 7),
            'severe': self.membership.triangular_membership(pain, 5, 8, 10),
            'extreme': self.membership.trapezoidal_membership(pain, 8, 9, 10, 10)
        }
    
    def breathing_fuzzification(self, breathing: float) -> Dict[str, float]:
        """Fuzzify breathing difficulty"""
        return {
            'normal': self.membership.trapezoidal_membership(breathing, 0, 0, 1, 2),
            'mild': self.membership.triangular_membership(breathing, 0, 3, 5),
            'moderate': self.membership.triangular_membership(breathing, 3, 6, 8),
            'severe': self.membership.trapezoidal_membership(breathing, 6, 8, 9, 9)
        }
    
    def consciousness_fuzzification(self, consciousness: float) -> Dict[str, float]:
        """Fuzzify consciousness level"""
        return {
            'alert': self.membership.trapezoidal_membership(consciousness, 0, 0, 1, 2),
            'drowsy': self.membership.triangular_membership(consciousness, 1, 4, 6),
            'confused': self.membership.triangular_membership(consciousness, 4, 7, 9),
            'unconscious': self.membership.trapezoidal_membership(consciousness, 7, 9, 10, 10)
        }
    
    def calculate_age_factor(self, age: int) -> float:
        """Calculate age-based risk factor"""
        if age < 2:
            return 1.3  # Infant
        elif age < 12:
            return 1.2  # Child
        elif age < 65:
            return 1.0  # Adult
        elif age < 80:
            return 1.1  # Senior
        else:
            return 1.2  # Elderly
    
    def calculate_triage_priority(self, symptoms: Dict[str, float], age: int) -> Dict:
        """Main fuzzy inference system for triage priority calculation"""
        
        # Extract and validate symptom values
        pain_level = max(0, min(10, float(symptoms.get('painLevel', 0))))
        breathing_diff = max(0, min(9, float(symptoms.get('breathingDifficulty', 0))))
        consciousness = max(0, min(10, float(symptoms.get('consciousnessLevel', 0))))
        chest_pain = max(0, min(10, float(symptoms.get('chestPain', 0))))
        heart_rate = max(0, min(9, float(symptoms.get('heartRate', 0))))
        bleeding = max(0, min(10, float(symptoms.get('bleeding', 0))))
        fever = max(0, min(7, float(symptoms.get('fever', 0))))
        headache = max(0, min(8, float(symptoms.get('headache', 0))))
        confusion = max(0, min(10, float(symptoms.get('confusion', 0))))
        
        # Fuzzify inputs
        pain_fuzzy = self.pain_fuzzification(pain_level)
        breathing_fuzzy = self.breathing_fuzzification(breathing_diff)
        consciousness_fuzzy = self.consciousness_fuzzification(consciousness)
        
        # Initialize urgency score
        urgency_score = 0
        
        # Critical rules (Priority 1 - Life threatening)
        if (consciousness_fuzzy['unconscious'] > 0.5 or 
            breathing_fuzzy['severe'] > 0.7 or 
            bleeding >= 10 or
            chest_pain >= 10):
            urgency_score = 95
            
        # High urgency rules (Priority 2 - Urgent)
        elif (consciousness_fuzzy['confused'] > 0.5 or
              breathing_fuzzy['moderate'] > 0.6 or
              pain_fuzzy['extreme'] > 0.6 or
              chest_pain >= 7 or
              heart_rate >= 9 or
              bleeding >= 6):
            urgency_score = 75
            
        # Moderate urgency (Priority 3 - Less urgent)
        elif (pain_fuzzy['severe'] > 0.5 or
              breathing_fuzzy['mild'] > 0.5 or
              fever >= 5 or
              headache >= 8 or
              confusion >= 7):
            urgency_score = 55
            
        # Low urgency (Priority 4 - Non-urgent)
        elif (pain_fuzzy['moderate'] > 0.5 or
              fever >= 2 or
              headache >= 5):
            urgency_score = 35
            
        # Minimal urgency (Priority 5 - Minimal)
        else:
            urgency_score = 15
        
        # Apply age factor
        age_factor = self.calculate_age_factor(age)
        urgency_score *= age_factor
        
        # Defuzzification - convert to priority level
        priority = 5
        priority_label = "MINIMAL - Within 120 min"
        estimated_wait_time = 120
        
        for p, threshold in self.priority_thresholds.items():
            if urgency_score >= threshold:
                priority = p
                break
        
        # Set priority labels and wait times
        priority_config = {
            1: ("CRITICAL - Immediate", 0),
            2: ("HIGH - Within 15 min", 15),
            3: ("MODERATE - Within 30 min", 30),
            4: ("LOW - Within 60 min", 60),
            5: ("MINIMAL - Within 120 min", 120)
        }
        
        priority_label, estimated_wait_time = priority_config[priority]
        
        return {
            'priority': priority,
            'priority_label': priority_label,
            'urgency_score': round(urgency_score, 2),
            'estimated_wait_time': estimated_wait_time
        }
    
    def add_patient(self, patient_data: Dict, symptoms: Dict) -> Patient:
        """Add new patient to the triage system"""
        
        # Calculate triage priority
        triage_result = self.calculate_triage_priority(symptoms, patient_data['age'])
        
        # Create patient object
        patient = Patient(
            id=self.patient_id_counter,
            name=patient_data['name'],
            age=patient_data['age'],
            gender=patient_data['gender'],
            symptoms=symptoms,
            additional_symptoms=patient_data.get('additionalSymptoms', ''),
            arrival_time=datetime.now(),
            priority=triage_result['priority'],
            priority_label=triage_result['priority_label'],
            urgency_score=triage_result['urgency_score'],
            estimated_wait_time=triage_result['estimated_wait_time']
        )
        
        # Insert patient in correct position based on priority and arrival time
        insert_index = len(self.patients)
        for i, existing_patient in enumerate(self.patients):
            if (existing_patient.priority > patient.priority or 
                (existing_patient.priority == patient.priority and 
                 existing_patient.arrival_time > patient.arrival_time)):
                insert_index = i
                break
        
        self.patients.insert(insert_index, patient)
        self.patient_id_counter += 1
        
        # Save to database
        self.save_patient_to_db(patient)
        
        logger.info(f"Patient {patient.name} added with priority {patient.priority}")
        return patient
    
    def save_patient_to_db(self, patient: Patient):
        """Save patient to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO patients (name, age, gender, symptoms, additional_symptoms,
                                    arrival_time, priority, priority_label, urgency_score, 
                                    estimated_wait_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient.name, patient.age, patient.gender, 
                json.dumps(patient.symptoms), patient.additional_symptoms,
                patient.arrival_time, patient.priority, patient.priority_label,
                patient.urgency_score, patient.estimated_wait_time
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving patient to database: {e}")
    
    def get_patients(self) -> List[Dict]:
        """Get all patients in queue"""
        return [patient.to_dict() for patient in self.patients]
    
    def get_statistics(self) -> Dict:
        """Get queue statistics"""
        total = len(self.patients)
        critical = len([p for p in self.patients if p.priority == 1])
        avg_wait = (sum(p.estimated_wait_time for p in self.patients) / total) if total > 0 else 0
        
        return {
            'total': total,
            'critical': critical,
            'avg_wait': round(avg_wait)
        }
    
    def remove_patient(self, patient_id: int) -> bool:
        """Remove patient from queue (when treated)"""
        for i, patient in enumerate(self.patients):
            if patient.id == patient_id:
                self.patients.pop(i)
                logger.info(f"Patient {patient.name} removed from queue")
                return True
        return False

class MedicalChatbot:
    """Intelligent medical chatbot for triage assistance"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self.responses = {
            'greetings': [
                "Hello! I'm here to help you with medical questions and triage information.",
                "Hi there! How can I assist you with your medical concerns today?",
                "Welcome! I'm your medical assistant AI. What would you like to know?"
            ],
            'triage_process': [
                "Triage is a process used in emergency rooms to prioritize patients based on the severity of their condition. We use 5 priority levels:",
                "Priority 1 (Critical): Immediate life-threatening conditions - seen immediately",
                "Priority 2 (High): Urgent but not immediately life-threatening - within 15 minutes", 
                "Priority 3 (Moderate): Serious but stable - within 30 minutes",
                "Priority 4 (Low): Less urgent conditions - within 60 minutes",
                "Priority 5 (Minimal): Non-urgent conditions - within 2 hours"
            ],
            'emergency_signs': [
                "🚨 Call 911 immediately if you experience any of these emergency warning signs:",
                "• Difficulty breathing or shortness of breath",
                "• Chest pain or pressure", 
                "• Severe bleeding that won't stop",
                "• Signs of stroke (sudden confusion, trouble speaking, face drooping)",
                "• Loss of consciousness",
                "• Severe allergic reactions",
                "• High fever with stiff neck",
                "• Severe abdominal pain"
            ],
            'symptoms': {
                'pain': "Pain levels help us prioritize care. Rate from 1-10: 1-3 is mild, 4-6 moderate, 7-8 severe, 9-10 excruciating. Severe pain (7+) gets higher priority.",
                'fever': "Fever indicates infection. Normal is 98.6°F. Low-grade: 99-101°F, High: 101-103°F, Very high: 103°F+. High fevers need prompt attention.",
                'breathing': "Breathing difficulty is serious. Any trouble breathing, especially at rest, needs immediate evaluation.",
                'chest': "Chest pain can indicate heart problems. Sharp, crushing, or radiating chest pain needs urgent care.",
                'bleeding': "Bleeding severity: Minor cuts heal naturally, moderate needs pressure/bandaging, severe/uncontrolled requires immediate care.",
                'headache': "Severe headaches, especially sudden onset or with fever/stiff neck, can indicate serious conditions."
            },
            'wait_times': "Wait times depend on your triage priority and current patient load. Critical patients are seen immediately, while less urgent cases may wait longer during busy periods.",
            'preparation': [
                "To help us serve you better, please prepare:",
                "• List of current medications",
                "• Insurance information and ID", 
                "• Recent medical history",
                "• Contact information for emergency contacts",
                "• Description of current symptoms and when they started"
            ]
        }
    
    def contains_emergency_keywords(self, message: str) -> bool:
        """Check if message contains emergency keywords"""
        emergency_keywords = [
            "can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath",
            "chest pain", "heart attack", "stroke", "unconscious", "severe bleeding",
            "allergic reaction", "overdose", "suicide", "choking"
        ]
        return any(keyword in message.lower() for keyword in emergency_keywords)
    
    def get_emergency_response(self, message: str) -> str:
        """Generate emergency response"""
        return ("🚨 EMERGENCY ALERT 🚨\n\n"
                "If you're experiencing a life-threatening emergency, please:\n"
                "• Call 911 immediately\n"
                "• Go to the nearest emergency room\n"
                "• Don't delay seeking immediate medical attention\n\n"
                "This chatbot is for information only and cannot replace emergency medical care.")
    
    def analyze_message(self, message: str) -> str:
        """Analyze user message and generate appropriate response"""
        msg = message.lower()
        
        # Emergency detection
        if self.contains_emergency_keywords(msg):
            return self.get_emergency_response(msg)
        
        # Symptom questions
        if 'pain' in msg or 'hurt' in msg:
            return self.responses['symptoms']['pain']
        if 'fever' in msg or 'temperature' in msg:
            return self.responses['symptoms']['fever']
        if 'breath' in msg or 'breathing' in msg:
            return self.responses['symptoms']['breathing']
        if 'chest' in msg:
            return self.responses['symptoms']['chest']
        if 'bleed' in msg or 'blood' in msg:
            return self.responses['symptoms']['bleeding']
        if 'headache' in msg or 'head' in msg:
            return self.responses['symptoms']['headache']
        
        # Process questions
        if 'triage' in msg or 'priority' in msg or ('how' in msg and 'work' in msg):
            return '\n\n'.join(self.responses['triage_process'])
        if 'emergency' in msg or 'urgent' in msg or 'serious' in msg:
            return '\n'.join(self.responses['emergency_signs'])
        if 'wait' in msg or 'time' in msg or 'long' in msg:
            return self.responses['wait_times']
        if 'prepare' in msg or 'bring' in msg or 'need' in msg:
            return '\n'.join(self.responses['preparation'])
        
        # Greetings
        if any(word in msg for word in ['hello', 'hi', 'help']):
            import random
            return random.choice(self.responses['greetings'])
        
        # Default response
        return ("I can help you with:\n"
                "• Understanding the triage process\n"
                "• Identifying emergency symptoms\n"
                "• Information about wait times\n"
                "• What to prepare for your visit\n"
                "• Symptom guidance\n\n"
                "What specific information would you like?")
    
    def save_chat_log(self, user_message: str, bot_response: str, session_id: str = None):
        """Save chat interaction to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO chat_logs (user_message, bot_response, session_id)
                VALUES (?, ?, ?)
            ''', (user_message, bot_response, session_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving chat log: {e}")
    
    def generate_response(self, message: str, session_id: str = None) -> str:
        """Generate response and log the interaction"""
        response = self.analyze_message(message)
        self.save_chat_log(message, response, session_id)
        return response

class TriageAnalytics:
    """Analytics and reporting for triage system"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
    
    def get_patient_history(self, limit: int = 100) -> List[Dict]:
        """Get patient history from database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM patients 
                ORDER BY arrival_time DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                patient_dict = dict(zip(columns, row))
                patient_dict['symptoms'] = json.loads(patient_dict['symptoms'])
                results.append(patient_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching patient history: {e}")
            return []
    
    def get_triage_statistics(self) -> Dict:
        """Get comprehensive triage statistics"""
        try:
            cursor = self.conn.cursor()
            
            # Overall statistics
            cursor.execute('SELECT COUNT(*) FROM patients')
            total_patients = cursor.fetchone()[0]
            
            # Priority distribution
            cursor.execute('''
                SELECT priority, COUNT(*) as count 
                FROM patients 
                GROUP BY priority
            ''')
            priority_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average urgency score by priority
            cursor.execute('''
                SELECT priority, AVG(urgency_score) as avg_score 
                FROM patients 
                GROUP BY priority
            ''')
            avg_scores = {row[0]: round(row[1], 2) for row in cursor.fetchall()}
            
            # Daily patient count (last 7 days)
            cursor.execute('''
                SELECT DATE(arrival_time) as date, COUNT(*) as count
                FROM patients 
                WHERE arrival_time >= datetime('now', '-7 days')
                GROUP BY DATE(arrival_time)
                ORDER BY date
            ''')
            daily_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_patients': total_patients,
                'priority_distribution': priority_dist,
                'average_scores': avg_scores,
                'daily_counts': daily_counts
            }
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}
    
    def generate_report(self) -> str:
        """Generate comprehensive triage report"""
        stats = self.get_triage_statistics()
        
        report = f"""
        TRIAGE SYSTEM ANALYTICS REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        OVERALL STATISTICS:
        Total Patients Processed: {stats.get('total_patients', 0)}
        
        PRIORITY DISTRIBUTION:
        """
        
        priority_labels = {
            1: "Critical",
            2: "High", 
            3: "Moderate",
            4: "Low",
            5: "Minimal"
        }
        
        for priority in range(1, 6):
            count = stats.get('priority_distribution', {}).get(priority, 0)
            avg_score = stats.get('average_scores', {}).get(priority, 0)
            label = priority_labels[priority]
            report += f"        Priority {priority} ({label}): {count} patients (Avg Score: {avg_score})\n"
        
        report += f"\n        RECENT ACTIVITY (Last 7 Days):\n"
        daily_counts = stats.get('daily_counts', {})
        for date, count in daily_counts.items():
            report += f"        {date}: {count} patients\n"
        
        return report

# Flask Web Application
app = Flask(__name__)
CORS(app)

# Initialize systems
triage_system = FuzzyTriageSystem()
chatbot = MedicalChatbot(triage_system.conn)
analytics = TriageAnalytics(triage_system.conn)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/triage', methods=['POST'])
def add_patient():
    """Add new patient to triage queue"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'age', 'gender']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        patient_data = {
            'name': data['name'],
            'age': int(data['age']),
            'gender': data['gender'],
            'additionalSymptoms': data.get('additionalSymptoms', '')
        }
        
        symptoms = data.get('symptoms', {})
        
        # Add patient to system
        patient = triage_system.add_patient(patient_data, symptoms)
        
        return jsonify({
            'success': True,
            'patient': patient.to_dict(),
            'message': f'Patient {patient.name} added with {patient.priority_label} priority'
        })
        
    except Exception as e:
        logger.error(f"Error adding patient: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queue', methods=['GET'])
def get_queue():
    """Get current patient queue"""
    try:
        patients = triage_system.get_patients()
        stats = triage_system.get_statistics()
        
        return jsonify({
            'success': True,
            'patients': patients,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error fetching queue: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>', methods=['DELETE'])
def remove_patient(patient_id):
    """Remove patient from queue"""
    try:
        success = triage_system.remove_patient(patient_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Patient removed successfully'})
        else:
            return jsonify({'error': 'Patient not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing patient: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = chatbot.generate_response(message, session_id)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get triage analytics"""
    try:
        stats = analytics.get_triage_statistics()
        return jsonify({
            'success': True,
            'analytics': stats
        })
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['GET'])
def generate_report():
    """Generate analytics report"""
    try:
        report = analytics.generate_report()
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get patient history"""
    try:
        limit = request.args.get('limit', 100, type=int)
        history = analytics.get_patient_history(limit)
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500

# Machine Learning Integration (Optional Enhancement)
class MLTriagePredictor:
    """Machine learning model for triage prediction enhancement"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = [
            'age', 'painLevel', 'breathingDifficulty', 'consciousnessLevel',
            'chestPain', 'heartRate', 'headache', 'confusion', 'bleeding', 'fever'
        ]
    
    def prepare_features(self, patient_data: Dict, symptoms: Dict) -> np.array:
        """Prepare features for ML model"""
        features = []
        features.append(patient_data['age'])
        
        for symptom in ['painLevel', 'breathingDifficulty', 'consciousnessLevel',
                       'chestPain', 'heartRate', 'headache', 'confusion', 
                       'bleeding', 'fever']:
            features.append(float(symptoms.get(symptom, 0)))
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data: List[Dict]):
        """Train ML model on historical data (placeholder for future enhancement)"""
        # This would implement actual ML training
        # For now, it's a placeholder showing the structure
        pass
    
    def predict_priority(self, patient_data: Dict, symptoms: Dict) -> int:
        """Predict triage priority using ML (placeholder)"""
        # This would use trained model for prediction
        # For now, returns None to fall back to fuzzy logic
        return None

# CLI Interface for System Management
def cli_interface():
    """Command line interface for system management"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python triage_system.py [command]")
        print("Commands:")
        print("  server - Start web server")
        print("  report - Generate analytics report")
        print("  test - Run test scenarios")
        return
    
    command = sys.argv[1]
    
    if command == 'server':
        print("Starting AI Hospital Triage System...")
        print("Server running on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    elif command == 'report':
        print("Generating Triage Analytics Report...")
        report = analytics.generate_report()
        print(report)
        
    elif command == 'test':
        print("Running Test Scenarios...")
        run_test_scenarios()
    
    else:
        print(f"Unknown command: {command}")

def run_test_scenarios():
    """Run test scenarios for system validation"""
    test_cases = [
        {
            'name': 'Critical Patient - Unconscious',
            'patient': {'name': 'John Doe', 'age': 45, 'gender': 'male'},
            'symptoms': {
                'painLevel': 8, 'breathingDifficulty': 3, 'consciousnessLevel': 10,
                'chestPain': 5, 'heartRate': 0, 'headache': 0, 'confusion': 0,
                'bleeding': 0, 'fever': 0
            }
        },
        {
            'name': 'High Priority - Chest Pain',
            'patient': {'name': 'Jane Smith', 'age': 60, 'gender': 'female'},
            'symptoms': {
                'painLevel': 7, 'breathingDifficulty': 6, 'consciousnessLevel': 0,
                'chestPain': 9, 'heartRate': 6, 'headache': 0, 'confusion': 0,
                'bleeding': 0, 'fever': 0
            }
        },
        {
            'name': 'Low Priority - Minor Symptoms',
            'patient': {'name': 'Bob Johnson', 'age': 30, 'gender': 'male'},
            'symptoms': {
                'painLevel': 3, 'breathingDifficulty': 0, 'consciousnessLevel': 0,
                'chestPain': 0, 'heartRate': 0, 'headache': 2, 'confusion': 0,
                'bleeding': 0, 'fever': 2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        patient = triage_system.add_patient(test_case['patient'], test_case['symptoms'])
        print(f"Result: Priority {patient.priority} - {patient.priority_label}")
        print(f"Urgency Score: {patient.urgency_score}/100")

if __name__ == '__main__':
    cli_interface()