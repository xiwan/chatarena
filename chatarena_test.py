from chatarena.arena import Arena

arena = Arena.from_config("examples/chameleon.json")
#arena.run(num_steps=50)
arena.launch_cli()

