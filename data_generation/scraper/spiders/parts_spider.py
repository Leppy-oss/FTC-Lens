import scrapy
from scrapy.loader import ItemLoader
from items import Product

class PartsSpider(scrapy.Spider):
	name = "parts"
	start_urls = [
		'https://www.gobilda.com/motion',
		'https://www.gobilda.com/structure/',
		'https://www.gobilda.com/electronics/'
	]

	def parse(self, response):
        # Get the link attribute from a product box
		partsList = response.css('li.product a')
		if partsList:
			# Sometimes, catalog pages are within others, so we have to recursively
			# go through each page to get to actual parts
			yield from response.follow_all(partsList, self.parse)
		else:
			# We are at an actual part page, with the step file,
			# so we process the actual values scraped from the page
			yield self.parse_product_page(response)

	def parse_product_page(self, response):
		step_file = response.css('a.ext-zip::attr(href)').get()
		name = response.css( 'h1.productView-title::text').get()
		if step_file and 'Bundle' not in name: # bundles will contain repeats, we don't want that
			loader = ItemLoader(Product(), response=response)
			loader.add_css('sku', 'span.productView-sku-input::text')
			loader.add_value('file_urls', [f'https://www.gobilda.com{step_file}'])
			loader.add_value('name', name)
			return loader.load_item()
		