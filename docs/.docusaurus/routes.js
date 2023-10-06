import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', 'f36'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', 'd25'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a7d'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c37'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '773'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '5b7'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '403'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog', '5ee'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive', '10f'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags', '9b3'),
    exact: true
  },
  {
    path: '/blog/tags/hello',
    component: ComponentCreator('/blog/tags/hello', 'a99'),
    exact: true
  },
  {
    path: '/blog/tags/textgrader',
    component: ComponentCreator('/blog/tags/textgrader', '1e2'),
    exact: true
  },
  {
    path: '/blog/welcome',
    component: ComponentCreator('/blog/welcome', '297'),
    exact: true
  },
  {
    path: '/docs/next',
    component: ComponentCreator('/docs/next', '03e'),
    routes: [
      {
        path: '/docs/next/backend/api',
        component: ComponentCreator('/docs/next/backend/api', 'da5'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/backend/congratulations',
        component: ComponentCreator('/docs/next/backend/congratulations', 'f81'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/backend/essay-grading',
        component: ComponentCreator('/docs/next/backend/essay-grading', 'f4d'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/backend/overview',
        component: ComponentCreator('/docs/next/backend/overview', '430'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/category/backend',
        component: ComponentCreator('/docs/next/category/backend', 'def'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/category/frontend',
        component: ComponentCreator('/docs/next/category/frontend', '871'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/frontend/congratulations',
        component: ComponentCreator('/docs/next/frontend/congratulations', 'dd2'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/frontend/essay-page',
        component: ComponentCreator('/docs/next/frontend/essay-page', 'b4f'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/frontend/layout',
        component: ComponentCreator('/docs/next/frontend/layout', 'f25'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/frontend/overview',
        component: ComponentCreator('/docs/next/frontend/overview', '5eb'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/next/intro',
        component: ComponentCreator('/docs/next/intro', '8d4'),
        exact: true,
        sidebar: "tutorialSidebar"
      }
    ]
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'd3b'),
    routes: [
      {
        path: '/docs/backend/api',
        component: ComponentCreator('/docs/backend/api', '124'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/backend/congratulations',
        component: ComponentCreator('/docs/backend/congratulations', '5e6'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/backend/essay-grading',
        component: ComponentCreator('/docs/backend/essay-grading', 'bcf'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/backend/overview',
        component: ComponentCreator('/docs/backend/overview', 'd2b'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/category/backend',
        component: ComponentCreator('/docs/category/backend', '240'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/category/frontend',
        component: ComponentCreator('/docs/category/frontend', 'd29'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/frontend/congratulations',
        component: ComponentCreator('/docs/frontend/congratulations', 'd3c'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/frontend/essay-page',
        component: ComponentCreator('/docs/frontend/essay-page', '50b'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/frontend/layout',
        component: ComponentCreator('/docs/frontend/layout', 'f63'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/frontend/overview',
        component: ComponentCreator('/docs/frontend/overview', 'fa2'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/intro',
        component: ComponentCreator('/docs/intro', 'fdc'),
        exact: true,
        sidebar: "tutorialSidebar"
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '1f9'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
