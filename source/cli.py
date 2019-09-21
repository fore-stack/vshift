# @Command-Line Interface
# Run on terminal to use the software
import click

@click.command()

@click.option('-a', '--analyze', nargs=2, help='[ Source Audio 🎧 ] [ Target Audio 🔊 ]')

@click.option('-c', '--convert', nargs=2, help='[ Conversion Matrices 🔢 ] [ Audio 🎧 ]')

@click.option('-p', '--prompt', is_flag=True, help='Use prompts for input instead.')

@click.option('-b', '--benchmark', is_flag=True, help='Use CMU Arctic Dataset.')

# @Click CLI-Interface
# accepts optionals (boolean) analyze, (boolean) convert, (boolean) prompt, and boolean(benchmark)
# does not return any value
def interface(analyze=True, convert=True, prompt=False, benchmark=False):

    import main 

    if benchmark:
        # @Benchmark Mode

        main.benchmark()

    elif prompt:
        # @Prompt Mode

        mode = click.prompt('Choose mode (analysis, conversion)', type=str)

        if mode == 'analysis':

            source = click.prompt('Enter source audio path 🎧 ', type=str)

            target = click.prompt('Enter target audio path 🔊 ', type=str)

            main.analyze(source, target)

        elif mode == 'conversion':

            matrices = click.prompt('Enter matrices path 🔢 ', type=str)

            original = click.prompt('Enter audio path 🎧 ', type=str)

            main.convert(matrices, original)

        else:

            click.secho('Error: Invalid Mode ✘', fg='red')

    else:

        # @Default Mode
        if len(analyze) == 2:

            main.analyze(analyze[0], analyze[1])
        
        elif len(convert) == 2:
            
            main.convert(convert[0], convert[1])

# @Run as standalone script
if __name__ == "__main__":

    interface()
    