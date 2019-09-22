new SmoothScroll('a[href*="#"]');

var termynal = new Termynal('.termynal',
    {
        lineData: [
            { type: 'input', value: 'python cli.py --prompt' },
            { type: 'input', prompt: 'Choose mode (analysis, conversion):', value: 'analysis'},
            { type: 'input', prompt: 'Enter source audio path 🎧 :', value: '/home/ralph/demo/parallel/cy.wav'},
            { type: 'input', prompt: 'Enter target audio path 🔊  :', value: '/home/ralph/demo/parallel/daniel.wav'},
            { value: 'Loading Audio Files: /home/ralph/demo/parallel/cy.wav /home/ralph/demo/parallel/daniel.wav 🔍 ' },
            { value: 'Training Gaussian Mixture Model 🎓' },
            { value: 'Training Finished on Gaussian Mixture Model (cy-daniel) ✓' }
        ]
    }
)