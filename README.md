# 🏈 FanDuel NFL DFS Optimizer

A powerful web-based Daily Fantasy Sports optimizer for FanDuel NFL contests, featuring advanced stacking strategies, defensive matchup targeting, and fantasy performance analysis.

## 🚀 Features

- **Advanced QB-WR/TE Stacking**: Enhanced multi-receiver stacking logic with tournament optimization
- **Defensive Matchup Targeting**: Attack the worst defenses for each position
- **Fantasy Performance Boosts**: 
  - WR: Targets, Receptions, FanDuel Points
  - RB: FanDuel Points, Attempts, Receptions
- **Tournament Strategy**: Optimized for 12-person league competition
- **Interactive Web Interface**: Easy-to-use Streamlit interface
- **Real-time Analytics**: Stacking analysis and performance metrics

## 📋 Requirements

Make sure you have the following files in your directory:
- `FanDuel-NFL-2025 EDT-10 EDT-05 EDT-121036-players-list (1).csv` (Your FanDuel player export)
- `NFL.xlsx` (Excel file with defensive stats, offensive stats, and fantasy performance data)

## 🛠️ Installation

1. Install Python (3.8 or higher)
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the App

### Local Development
```bash
streamlit run dfs_optimizer_app.py
```

### Sharing with Others

#### Option 1: Streamlit Cloud (Recommended)
1. Upload your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy your app

#### Option 2: Local Network Sharing
```bash
streamlit run dfs_optimizer_app.py --server.address 0.0.0.0
```
Then share your local IP address with others on your network.

#### Option 3: Cloud Deployment
Deploy to platforms like:
- Heroku
- Google Cloud Platform
- AWS
- Azure

## ⚙️ Configuration

### Optimization Settings
- **Number of Simulations**: 1,000 - 20,000 (default: 10,000)
- **Stacking Probability**: 0% - 100% (default: 55%)
- **Elite Target Boost**: 0% - 100% (default: 45%)
- **Great Target Boost**: 0% - 100% (default: 25%)

### Display Settings
- **Number of Top Lineups**: 5 - 50 (default: 20)

## 📊 Data Sources Required

1. **FanDuel Player List**: Export from FanDuel contest
2. **NFL.xlsx** with these sheets:
   - `Defense Data 2025`: Defensive statistics
   - `Offense Data 2025`: Offensive statistics  
   - `Fantasy`: Fantasy performance data
   - `Teams`: Team name mappings

## 🎯 Strategy Features

- **Tournament Optimization**: Reduced stack frequency (55%) for contrarian lineups
- **Multi-Receiver Stacking**: 65% chance for QB+2+ receivers, 85% for QB+1+ receivers
- **Enhanced Targeting**: 45% boost for elite matchups, 25% for great matchups
- **Value-Based Selection**: Prioritizes high-volume, productive players
- **Salary Optimization**: Smart defense selection to maximize skill position budget

## 📈 Analytics

The app provides:
- Lineup generation success rates
- Stacking distribution analysis
- Average vs. best projected points
- Fantasy performance boost effectiveness
- Matchup quality breakdown

## 🔧 Troubleshooting

**File Not Found Errors**: Ensure all required data files are in the same directory as the app

**Performance Issues**: Reduce number of simulations for faster results

**Memory Issues**: Use smaller datasets or reduce simulation count

## 🤝 Contributing

Feel free to enhance the optimizer with additional features:
- More advanced stacking strategies
- Additional data sources
- Machine learning predictions
- Custom scoring systems

## 📄 License

This project is for educational and personal use. Please comply with FanDuel's terms of service when using this tool.