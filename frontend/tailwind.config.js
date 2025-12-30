/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'sans': ['"DM Sans"', 'system-ui', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        // Deep slate base
        surface: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          800: '#1e293b',
          850: '#172033',
          900: '#0f172a',
          950: '#080d19',
        },
        // Vibrant teal accent
        accent: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
        },
        // Risk semantic colors
        risk: {
          critical: '#f43f5e',
          high: '#fb923c',
          medium: '#fbbf24',
          low: '#34d399',
        },
      },
      boxShadow: {
        'glow': '0 0 20px -5px rgba(20, 184, 166, 0.4)',
        'glow-sm': '0 0 10px -3px rgba(20, 184, 166, 0.3)',
        'card': '0 4px 20px -4px rgba(0, 0, 0, 0.5)',
        'card-hover': '0 8px 30px -4px rgba(0, 0, 0, 0.6)',
      },
      backgroundImage: {
        'gradient-accent': 'linear-gradient(135deg, #14b8a6 0%, #2dd4bf 100%)',
        'gradient-danger': 'linear-gradient(135deg, #f43f5e 0%, #fb7185 100%)',
        'gradient-warning': 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)',
        'gradient-success': 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
        'gradient-surface': 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 15px -5px rgba(20, 184, 166, 0.3)' },
          '100%': { boxShadow: '0 0 25px -5px rgba(20, 184, 166, 0.5)' },
        },
      },
    },
  },
  plugins: [],
}
