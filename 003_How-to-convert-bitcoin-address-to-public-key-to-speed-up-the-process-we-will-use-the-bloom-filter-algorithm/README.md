# How to convert Bitcoin address to public key to speed up the process we will use the Bloom filter algorithm

<!-- wp:image {"lightbox":{"enabled":false},"id":2516,"sizeSlug":"large","linkDestination":"custom","align":"center"} -->
<figure class="wp-block-image aligncenter size-large"><a href="https://cryptodeeptech.ru/discrete-logarithm" target="_blank" rel=" noreferrer noopener"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-2-1-1024x787.png" alt="" class="wp-image-2516"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>It is not possible to convert a Bitcoin address to a public key directly online without access to the specific transactions that the address was involved in. However, if you have access to the blockchain and can extract the transactions that the address was involved in, you can attempt to extract the public key from the signatures of those transactions.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>To speed up the process, you can use the Bloom filter algorithm, but its application in this case is more complex and requires a deep understanding of the blockchain and cryptography. Below is a simplified example of how you can try to extract the public key from transactions, but this is not a simple task and requires access to specific blockchain data.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Example Python script to extract public key</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>This script cannot be run directly online without access to the blockchain and specific transactions. It is intended to demonstrate the concept.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import requests
import hashlib

def get_transaction_data(address):
    <em># Получаем данные транзакций для адреса</em>
    url = f'https://blockchain.info/q/multiaddr?active={address}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

def extract_public_key(tx_data):
    <em># Извлечение открытого ключа из транзакций</em>
    <em># Это упрощенный пример и не работает напрямую</em>
    <em># Для реализации нужно иметь доступ к конкретным подписям транзакций</em>
    <em># и использовать криптографические библиотеки для извлечения ключа</em>
    pass

def main():
    address = input("Введите адрес биткойна: ")
    tx_data = get_transaction_data(address)
    
    if tx_data:
        print("Данные транзакций получены.")
        <em># Здесь нужно реализовать логику для извлечения открытого ключа</em>
        <em># из подписей транзакций, что требует доступа к блокчейну</em>
        <em># и криптографических библиотек.</em>
    else:
        print("Не удалось получить данные транзакций.")

if __name__ == "__main__":
    main()
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Using the Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>A bloom filter is a data structure that can quickly determine whether an element is present in a data set. However, for extracting the public key from transactions, this is not a straightforward solution. Instead, you can use it to filter transactions that potentially contain the public key, but this will require significant changes to the script and a deep understanding of the blockchain.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Important Notes</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Blockchain Access</strong> : To implement this functionality, you will need direct access to the blockchain data or use an API that provides access to transaction signatures.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Cryptographic libraries</strong> : You will need to use cryptographic libraries (such as <code>ecdsa</code>, <code>cryptography</code>) to work with signatures and extract the public key.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Difficulty of the Problem</strong> : Extracting public keys from transactions is a complex problem that requires a deep understanding of cryptography and blockchain.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>In real life, specialized tools and libraries such as&nbsp;<code>bitcoin-core</code>or are used to solve such problems&nbsp;<code>pycryptodome</code>, which provide the necessary functions for working with cryptography and blockchain.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Which Python libraries are best to use for working with Bitcoin</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To work with Bitcoin in Python, you can use the following libraries:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>bitcoinlib</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : This is a powerful library that offers a wide range of tools for working with the Bitcoin blockchain. It allows you to create different types of wallets, interact with the blockchain, create, sign and validate transactions, and also supports working with Bitcoin Script <a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/" target="_blank" rel="noreferrer noopener">1 </a><a href="https://ru.tradingview.com/news/bitsmedia:62a540b0167b8:0/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Features</strong> : Create wallets, generate bitcoin addresses, work with transactions.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>pycryptodome</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Although not specifically mentioned in search results, this library is useful for cryptographic operations that may be needed when working with Bitcoin.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Features</strong> : Cryptographic functions including encryption and hashing.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>blockchain</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : This library allows you to interact with the Bitcoin blockchain, get information about transactions and addresses <a href="https://habr.com/ru/articles/525638/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Features</strong> : Obtaining transaction and address data.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>block-io</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Also mentioned as useful for working with Bitcoin, although there are no details about it in the search results <a href="https://habr.com/ru/articles/525638/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Features</strong> : Interaction with blockchain.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These libraries can help develop Bitcoin-related applications, including creating wallets, analyzing transactions, and interacting with the blockchain. More complex tasks, such as extracting public keys from transactions, may require additional cryptographic processing.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What are some alternatives to Bloom filter to speed up work with Bitcoin addresses?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To speed up work with Bitcoin addresses, in addition to Bloom filters, you can use the following alternative methods and data structures:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Hash tables</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : Hash tables allow you to quickly store and retrieve data by key. They can be used to quickly search and filter Bitcoin addresses.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Fast search and insertion of data, low probability of collisions with the correct choice of hash functions.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Three</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : A tree (or prefix tree) is a data structure that allows strings to be stored and retrieved efficiently by prefix. This can be useful for filtering addresses by specific prefixes.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Efficient storage and retrieval of strings with common prefixes.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Cuckoo Filter</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : The Cuckoo filter is a probabilistic data structure that can be used to quickly determine whether an element belongs to a set. It is more efficient than the Bloom filter in terms of false positives.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Low probability of false positives compared to Bloom filter.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>B-trees</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : B-trees are balanced trees that can be used to efficiently store and search data in large sets. They are especially useful in databases.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Efficient search and insertion of data in large data sets.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Hyperloglog</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Description</strong> : A hyperloglog is a probabilistic data structure that allows one to estimate the number of unique elements in a data set. Although not a direct alternative for address filtering, it can be useful for analyzing data sets.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Advantages</strong> : Efficient estimation of the number of unique elements.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">An example implementation of a hash table for working with bitcoin addresses</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>class BitcoinAddressTable:
    def __init__(self):
        self.table = {}

    def add_address(self, address):
        self.table[address] = True

    def check_address(self, address):
        return address in self.table

<em># Пример использования</em>
table = BitcoinAddressTable()
table.add_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
print(table.check_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"))  <em># True</em>
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>This example demonstrates a simple implementation of a hash table for quickly storing and verifying Bitcoin addresses. More complex tasks, such as filtering by prefixes or using other data structures, will require a more complex implementation.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://dzen.ru/a/YsR8D0FFcSzXhY9v">https://dzen.ru/a/YsR8D0FFcSzXhY9v</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.weblancer.net/freelance/prikladnoe-po-23/generatsiya-bitkoin-adresa-na-python-s-ispolzovaniem-gpu-1221160/">https://www.weblancer.net/freelance/prikladnoe-po-23/generatsiya-bitkoin-adresa-na-python-s-ispolzovaniem-gpu-1221160/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/">https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://cryptodeep.ru/rowhammer-attack/">https://cryptodeep.ru/rowhammer-attack/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forum.bits.media/index.php?%2Fblogs%2Fentry%2F3154-%D1%81%D0%B0%D0%BC%D0%B0%D1%8F-%D0%BF%D0%B5%D1%80%D0%B2%D0%B0%D1%8F-%D1%81%D0%B5%D1%80%D1%8C%D0%B5%D0%B7%D0%BD%D0%B0%D1%8F-%D1%83%D1%8F%D0%B7%D0%B2%D0%B8%D0%BC%D0%BE%D1%81%D1%82%D1%8C-%D0%B2-blockchain-%D0%B8-%D0%BA%D0%B0%D0%BA-%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B8%D1%82%D1%8C-%D0%BF%D1%83%D0%B1%D0%BB%D0%B8%D1%87%D0%BD%D1%8B%D0%B9-%D0%BA%D0%BB%D1%8E%D1%87-bitcoin-ecdsa-%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-rsz-%D0%B8%D0%B7-%D1%84%D0%B0%D0%B9%D0%BB%D0%B0-rawtx%2F">https://forum.bits.media/index.php?%2Fblogs%2Fentry%2F3154-%D1%81%D0%B0%D0%BC%D0%B0%D1%8F-%D0%BF%D0%B5%D1%80%D0%B2%D0%B0%D1%8F-%D1%81 %D0%B5%D1%80%D1%8C%D0%B5%D0%B7%D0%BD%D0%B0%D1%8F-%D1%83%D1%8F%D0%B7 %D0%B2%D0%B8%D0%BC%D0%BE%D1%81%D1%82%D1%8C-%D0%B2-blockchain-%D0%B8 -%D0%BA%D0%B0%D0%BA-%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B8%D1%82%D1%8 C-%D0%BF%D1%83%D0%B1%D0%BB%D0%B8%D1%87%D0%BD%D1%8B%D0%B9-%D0%BA%D0% BB%D1%8E%D1%87-bitcoin-ecdsa-%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D 0%B8%D0%B5-rsz-%D0%B8%D0%B7-%D1%84%D0%B0%D0%B9%D0%BB%D0%B0-rawtx%2F</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/564256/">https://habr.com/ru/articles/564256/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ftp.zhirov.kz/books/IT/Other/%D0%A0%D0%B5%D0%B0%D0%B3%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%20%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B8%D0%BD%D1%86%D0%B8%D0%B4%D0%B5%D0%BD%D1%82%D1%8B%20(%D0%A1%D1%82%D0%B8%D0%B2%20%D0%AD%D0%BD%D1%81%D0%BE%D0%BD).pdf">https://ftp.zhirov.kz/books/IT/Other/%D0%A0%D0%B5%D0%B0%D0%B3%D0%B8%D1%80%D0%BE% D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%20%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1% 8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B8%D0%BD%D1%86%D0%B8%D0%B4%D0%B5%D0 %BD%D1%82%D1%8B%20(%D0%A1%D1%82%D0%B8%D0%B2%20%D0%AD%D0%BD%D1%81%D0%BE%D0%BD).pdf</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82">https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/">https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.tradingview.com/news/bitsmedia:62a540b0167b8:0/">https://ru.tradingview.com/news/bitsmedia:62a540b0167b8:0/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://proglib.io/p/kak-python-primenyaetsya-v-blokcheyn-2021-03-19">https://proglib.io/p/kak-python-primenyaetsya-v-blokcheyn-2021-03-19</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-blokchejn/">https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-blokchejn/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.bits.media/kod-dlya-koda-kak-yazyk-programmirovaniya-python-primenyaetsya-v-kriptoindustrii/">https://www.bits.media/kod-dlya-koda-kak-yazyk-programmirovaniya-python-primenyaetsya-v-kriptoindustrii/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://blog.skillfactory.ru/top-29-bibliotek-dlya-python-chem-polzuyutsya-razrabotchiki/">https://blog.skillfactory.ru/top-29-bibliotek-dlya-python-chem-polzuyutsya-razrabotchiki/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.stackoverflow.com/questions/1157945/%D0%9F%D0%BE%D0%B4%D1%81%D0%BA%D0%B0%D0%B6%D0%B8%D1%82%D0%B5-python-%D0%B1%D0%B8%D0%B1%D0%BB%D0%B8%D0%BE%D1%82%D0%B5%D0%BA%D1%83-%D0%B4%D0%BB%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B-%D1%81-etherium">https://ru.stackoverflow.com/questions/1157945/%D0%9F%D0%BE%D0%B4%D1%81%D0%BA%D0%B0%D0%B6%D0%B8%D1%82%D0%B5-python-%D0%B1%D 0%B8%D0%B1%D0%BB%D0%B8%D0%BE%D1%82%D0%B5%D0%BA%D1%83-%D0%B4%D0%BB%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B-%D1%81-etherium</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/491132/">https://habr.com/ru/articles/491132/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forklog.com/exclusive/bitkoin-koshelki-sravnili-po-48-kriteriyam-najdite-svoj">https://forklog.com/exclusive/bitkoin-koshelki-sravnili-po-48-kriteriyam-najdite-svoj</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://nuancesprog.ru/p/21154/">https://nuancesprog.ru/p/21154/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/438336/">https://habr.com/ru/articles/438336/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.osp.ru/os/2023/02/13057166">https://www.osp.ru/os/2023/02/13057166</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forum.bits.media/index.php?%2Ftopic%2F174462-%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D1%8B-%D0%B4%D0%BB%D1%8F-%D0%B2%D1%8B%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F-%D0%BF%D1%80%D0%B8%D0%B2%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE-%D0%BA%D0%BB%D1%8E%D1%87%D0%B0%2Fpage%2F15%2F">https://forum.bits.media/index.php?%2Ftopic%2F174462-%D0%BF%D1%80%D0%BE%D 0%B3%D1%80%D0%B0%D0%BC%D0%BC%D1%8B-%D0%B4%D0%BB%D1%8F-%D0%B2%D1%8B%D1%87%D 0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F-%D0%BF%D1%80%D0%B8%D0%B2%D0%B0%D 1%82%D0%BD%D0%BE%D0%B3%D0%BE-%D0%BA%D0%BB%D1%8E%D1%87%D0%B0%2Fpage%2F15%2F</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong><a href="https://github.com/demining/CryptoDeepTools/tree/main/37DiscreteLogarithm">Source code</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://t.me/cryptodeeptech">Telegram: https://t.me/cryptodeeptech</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://youtu.be/i9KYih_ffr8">Video: https://youtu.be/i9KYih_ffr8</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://dzen.ru/video/watch/6784be61b09e46422395c236">Video tutorial: https://dzen.ru/video/watch/6784be61b09e46422395c236</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://cryptodeeptech.ru/discrete-logarithm">Source: https://cryptodeeptech.ru/discrete-logarithm</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading" id="user-content-block-67e26253-470e-4432-a4e1-65b7b8b74c1b">Useful information for enthusiasts:</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><a href="https://github.com/demining/bitcoindigger/blob/main/001_Creating_RawTX_Bitcoin_Transactions_Using_Bloom_Filter_in_Python/README.md#useful-information-for-enthusiasts"></a></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul id="user-content-block-5e543c86-afad-430c-8aac-8ff0ffccb4e2" class="wp-block-list"><!-- wp:list-item -->
<li><strong>[1]  </strong><em><strong><a href="https://www.youtube.com/@cryptodeeptech">YouTube Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[2]  </strong><em><strong><a href="https://t.me/s/cryptodeeptech">Telegram Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[3]  </strong><a href="https://github.com/demining/CryptoDeepTools"><em><strong>GitHub Repositories  </strong></em> </a><em><strong><a href="https://github.com/demining/CryptoDeepTools">CryptoDeepTools</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[4]   </strong><em><strong><a href="https://t.me/ExploitDarlenePRO">Telegram: ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[5]  </strong><em><strong><a href="https://www.youtube.com/@ExploitDarlenePRO">YouTube Channel ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[6]  </strong><em><strong><a href="https://github.com/keyhunters">GitHub Repositories Keyhunters</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[7]  </strong><em><strong><a href="https://t.me/s/Bitcoin_ChatGPT">Telegram: Bitcoin ChatGPT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[8]  </strong><strong><em><a href="https://www.youtube.com/@BitcoinChatGPT">YouTube Channel BitcoinChatGPT</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[9]  </strong><a href="https://bitcoincorewallet.ru/"><strong><em>Bitcoin Core Wallet Vulnerability</em></strong></a><a href="https://bitcoincorewallet.ru/"> </a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[10]  </strong> <strong><a href="https://btcpays.org/"><em>BTC PAYS DOCKEYHUNT</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[11]   </strong><em><strong><a href="https://dockeyhunt.com/"> DOCKEYHUNT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[12]   </strong><em><strong><a href="https://t.me/s/DocKeyHunt">Telegram: DocKeyHunt</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[13]   </strong><em><strong><a href="https://exploitdarlenepro.com/">ExploitDarlenePRO.com</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[14]  </strong><em><strong><a href="https://github.com/demining/Dust-Attack">DUST ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[15]  </strong><em><strong><a href="https://bitcoin-wallets.ru/">Vulnerable Bitcoin Wallets</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[16]  </strong> <em><strong><a href="https://www.youtube.com/playlist?list=PLmq8axEAGAp_kCzd9lCjX9EabJR9zH3J-">ATTACKSAFE SOFTWARE</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[17]  </strong><em><strong><a href="https://youtu.be/CzaHitewN-4"> LATTICE ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[18]   </strong><em><strong><a href="https://github.com/demining/Kangaroo-by-JeanLucPons"> RangeNonce</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[19]   <em><a href="https://bitcoinwhoswho.ru/">BitcoinWhosWho</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[20]   <em><a href="https://coinbin.ru/">Bitcoin Wallet by Coinbin</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[21]  </strong><em><strong><a href="https://cryptodeeptech.ru/polynonce-attack/">POLYNONCE ATTACK</a></strong></em><em><strong> <a href="https://cryptodeeptech.ru/polynonce-attack/"></a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[22]  </strong> <a href="https://cold-wallets.ru/"><strong><em>Cold Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[23]  </strong> <a href="https://bitcointrezor.ru/"><strong><em>Trezor Hardware Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[24]   <a href="https://bitcoinexodus.ru/"><em>Exodus Wallet Vulnerability</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[25]  <em><a href="https://bitoncoin.org/">BITCOIN DOCKEYHUNT</a></em><em> <a href="https://bitoncoin.org/"></a></em></strong></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:image {"linkDestination":"custom"} -->
<figure class="wp-block-image"><a href="https://cryptodeeptech.ru/discrete-logarithm/" target="_blank" rel="noreferrer noopener"><img src="https://camo.githubusercontent.com/3568257c92c143826ea99b9e75ccdf3b05ed58e05081f33d93b5aca5129d67fc/68747470733a2f2f63727970746f64656570746f6f6c2e72752f77702d636f6e74656e742f75706c6f6164732f323032342f31322f474f4c4431303331422d31303234783537362e706e67" alt="Discrete Logarithm mathematical methods and tools for recovering cryptocurrency wallets Bitcoin"/></a></figure>
<!-- /wp:image -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><a href="https://www.facebook.com/sharer.php?u=https%3A%2F%2Fpolynonce.ru%2F%25d0%25ba%25d0%25b0%25d0%25ba-%25d0%25ba%25d0%25be%25d0%25bd%25d0%25b2%25d0%25b5%25d1%2580%25d1%2582%25d0%25b8%25d1%2580%25d0%25be%25d0%25b2%25d0%25b0%25d1%2582%25d1%258c-%25d0%25b1%25d0%25b8%25d1%2582%25d0%25ba%25d0%25be%25d0%25b8%25d0%25bd-%25d0%25b0%25d0%25b4%25d1%2580%25d0%25b5%25d1%2581-%25d0%25b2-%25d0%25bf%25d1%2583%2F" target="_blank" rel="noreferrer noopener"></a></p>
<!-- /wp:paragraph -->
