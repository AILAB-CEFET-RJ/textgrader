import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/undraw_textgrader_use.svg').default,
    description: (
      <>
        Textgrader was designed from the ground up to be easily installed and
        used to get your project up and running quickly.
      </>
    ),
  },
  {
    title: 'Focus on What Matters',
    Svg: require('@site/static/img/undraw_textgrader_write.svg').default,
    description: (
      <>
        Textgrader lets you focus on your essay, and we&apos;ll do the chores. Go
        ahead and write your essay on <code>/redacao</code> path.
      </>
    ),
  },
  {
    title: 'Powered by React and FastAPI',
    Svg: require('@site/static/img/undraw_textgrader_code.svg').default,
    description: (
      <>
        Extend or customize your website layout by reusing React. Extend your API by using FastAPI.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
