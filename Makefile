config:
	@echo config bundle local path: .bundle/gems
	@bundle config set --local path .bundle/gems

preview:
	@bundle exec jekyll serve --incremental

