import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'mya-saffron': '#F6B23E',
        'mya-indigo': '#302B63',
        'mya-purple': '#6C4AB6'
      },
      fontFamily: {
        sans: ['"SF Pro Text"', '"Segoe UI"', 'Inter', 'system-ui', 'sans-serif']
      }
    }
  },
  plugins: []
};

export default config;
