BOT_NAME = 'ftc_lens_scraper'

SPIDER_MODULES = ['spiders']
NEWSPIDER_MODULE = 'spiders'

ROBOTSTXT_OBEY = True

ITEM_PIPELINES = {
    'pipelines.FtcLensScraperPipeline': 300,
	'scrapy.pipelines.files.FilesPipeline': 1
}

FILES_STORE = '../scraped_models'