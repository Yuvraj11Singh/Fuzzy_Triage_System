# 🏥 AI Hospital Triage System

An advanced web-based hospital triage system that uses fuzzy logic algorithms to prioritize patients based on symptom severity and medical urgency.

## ✨ Features

### 🔬 Advanced AI Triage Assessment
- **Fuzzy Logic Algorithm**: Sophisticated patient prioritization using triangular, trapezoidal, and sigmoid membership functions
- **Multi-factor Analysis**: Considers age, symptoms severity, vital signs, and medical history
- **5-Level Priority System**: From critical (immediate) to minimal (2+ hours wait time)
- **Dynamic Risk Assessment**: Real-time identification of critical conditions

### 📊 Real-time Queue Management
- **Live Dashboard**: Monitor patient queue with real-time statistics
- **Smart Queue Insertion**: Automatic patient ordering based on priority and urgency scores
- **Dynamic Wait Times**: Continuously updated estimated wait times
- **Critical Alerts**: Immediate notifications for life-threatening conditions

### 🤖 Intelligent Medical Assistant
- **24/7 AI Chatbot**: Provides medical guidance and triage information
- **Emergency Detection**: Automatically identifies emergency keywords and provides immediate guidance
- **Contextual Responses**: Learns from conversation history for better assistance
- **Quick Actions**: Predefined buttons for common medical questions

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Glass Morphism**: Modern backdrop blur effects and translucent panels
- **Smooth Animations**: CSS transitions and loading states for better user experience
- **Accessibility**: WCAG compliant design with proper contrast and navigation

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No server setup required - runs entirely in the browser

### Installation
1. Clone or download the repository
```bash
git clone https://github.com/Yuvraj11Singh/Sample.git
cd Sample/Hospital_Triage
```

2. Open `part3.html` in your web browser
```bash
# On macOS
open part3.html

# On Windows
start part3.html

# On Linux
xdg-open part3.html
```

## 📖 How to Use

### 1. Patient Triage Assessment
1. **Patient Information**: Enter name, age, and gender
2. **Symptom Assessment**: Use the intuitive sliders and dropdowns to assess:
   - Primary symptoms (pain level, breathing, consciousness)
   - Cardiovascular symptoms (chest pain, heart rhythm)
   - Neurological symptoms (headache, confusion)
   - Critical signs (bleeding, temperature)
3. **Additional Information**: Add medical history and other symptoms
4. **Submit**: Get instant AI-powered priority assessment

### 2. Queue Management
- **View Statistics**: Monitor total patients, critical cases, and average wait times
- **Patient Queue**: See all patients ordered by priority with estimated wait times
- **Critical Alerts**: Receive immediate notifications for critical patients

### 3. Medical Assistant
- **Ask Questions**: Chat with the AI about symptoms, procedures, and medical guidance
- **Quick Actions**: Use predefined buttons for common topics
- **Emergency Help**: Get immediate guidance for emergency situations

## 🧠 Triage Algorithm

### Priority Levels
| Priority | Label | Wait Time | Description |
|----------|-------|-----------|-------------|
| 1 | Critical - Immediate | 0 min | Life-threatening conditions |
| 2 | High - Within 15 min | 15 min | Urgent conditions |
| 3 | Moderate - Within 30 min | 30 min | Serious but stable |
| 4 | Low - Within 60 min | 60 min | Less urgent conditions |
| 5 | Minimal - Within 120 min | 120 min | Non-urgent conditions |

### Assessment Factors
- **Symptom Severity**: Pain level, breathing difficulty, consciousness level
- **Age Factor**: Enhanced priority for pediatric and geriatric patients
- **Vital Signs**: Heart rate, blood pressure, temperature
- **Critical Conditions**: Bleeding, chest pain, neurological symptoms
- **Risk Factors**: Medical history and comorbidities

### Fuzzy Logic Implementation
The system uses advanced fuzzy logic with:
- **Triangular Membership Functions**: For symptom severity assessment
- **Trapezoidal Membership Functions**: For boundary conditions
- **Sigmoid Functions**: For smooth transitions between priority levels
- **Rule-based Inference**: Combines multiple symptoms for final assessment

## 🛠️ Technical Details

### Technologies Used
- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with CSS Grid, Flexbox, and custom properties
- **Vanilla JavaScript**: No external dependencies for maximum performance
- **ES6+ Features**: Classes, arrow functions, async/await, destructuring

### Browser Compatibility
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

### Performance Features
- **Lightweight**: No external frameworks or libraries
- **Fast Loading**: Optimized CSS and JavaScript
- **Smooth Animations**: Hardware-accelerated CSS transitions
- **Responsive**: Adaptive layout for all screen sizes

## 📱 Responsive Design

The system is fully responsive and adapts to:
- **Desktop**: Full three-panel layout with complete functionality
- **Tablet**: Optimized layout with touch-friendly controls
- **Mobile**: Single-column layout with swipe navigation

## 🔒 Privacy & Security

- **Client-side Only**: All processing happens in the browser
- **No Data Storage**: Patient information is not stored or transmitted
- **HIPAA Considerations**: Designed with healthcare privacy in mind
- **Secure**: No external API calls or data transmission

## 🎯 Use Cases

### Emergency Departments
- Rapid patient assessment and prioritization
- Reduce waiting room overcrowding
- Improve patient flow and satisfaction
- Support medical staff decision-making

### Urgent Care Centers
- Standardized triage protocols
- Consistent patient assessment
- Educational tool for staff training
- Quality improvement initiatives

### Medical Education
- Training tool for healthcare students
- Demonstration of triage principles
- Interactive learning experience
- Case study simulations

## 🚨 Important Disclaimers

⚠️ **Medical Disclaimer**: This system is for educational and demonstration purposes only. It should not be used for actual medical decision-making without proper medical oversight.

⚠️ **Emergency**: In case of life-threatening emergencies, always call 911 or go directly to the nearest emergency room.

⚠️ **Professional Judgment**: This tool is designed to support, not replace, professional medical judgment and clinical decision-making.

## 📈 Future Enhancements

- [ ] Integration with Electronic Health Records (EHR)
- [ ] Machine Learning model training on historical data
- [ ] Multi-language support for diverse populations
- [ ] Voice input for hands-free operation
- [ ] Integration with hospital management systems
- [ ] Advanced analytics and reporting features
- [ ] Telemedicine integration capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is available under the MIT License. See the LICENSE file for more details.

## 👥 Authors

- **Yuvraj Singh** - Initial development and design

## 🙏 Acknowledgments

- Medical professionals who provided input on triage protocols
- Open source community for inspiration and best practices
- Healthcare institutions for real-world requirements and feedback

---

### 📞 Support

For questions, issues, or support, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation and FAQ

**Remember**: This is a demonstration system. Always consult with qualified medical professionals for actual healthcare decisions.
