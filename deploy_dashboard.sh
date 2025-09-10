#!/bin/bash

# Data Science Dashboard Deployment Script
# Fixes Vercel project naming issue and deploys dashboard

echo "🚀 Deploying Data Science Dashboard to Vercel..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "index.html" ] || [ ! -f "vercel.json" ]; then
    echo "❌ Error: Please run this script from the dashboard directory"
    echo "   cd dashboard"
    echo "   ./deploy_dashboard.sh"
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found."
    echo "📦 Install with: npm install -g vercel"
    exit 1
fi

echo "📋 Using project name: data-science-dashboard"
echo "   (Vercel requires lowercase names)"
echo ""

# Deploy to production
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your dashboard will be available at the URL shown above"
echo ""
echo "💡 Tip: Bookmark this URL to check student progress anytime!"