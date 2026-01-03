import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'retro-bg': '#0b0b0c',
        'retro-surface': '#202123',
        'retro-border': '#3d3f45',
        'retro-title-active': '#2d2f36',
        'retro-text': '#e8e8e8',
        'retro-accent': '#10a37f'
      },
      fontFamily: {
        sans: [
          '"MS Gothic"',
          '"Fixedsys"',
          '"Lucida Console"',
          'monospace'
        ]
      }
    }
  },
  plugins: []
};

export default config;
