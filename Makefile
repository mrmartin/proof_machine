.PHONY: build test demo clean

build:
	dune build

test: build
	dune runtest
	bash tests/e2e.sh

demo: build
	bash tests/e2e.sh

clean:
	dune clean
	rm -f examples/*/*.kf examples/*/*.cert examples/*/*.tex
	rm -f examples/*/*.aux examples/*/*.log examples/*/*.pdf
