var filterYearStartList = document.getElementById("filter-year-start");
var filterYearStartSel = filterYearStartList.options[filterYearStartList.selectedIndex].value;
var filterYearEndList = document.getElementById("filter-year-end");
var filterYearEndSel = filterYearEndList.options[filterYearEndList.selectedIndex].value;
var xAxisData = [];

for (var j = filterYearStartSel; j <= filterYearEndSel; j++) {
    xAxisData.push(j);
}

var word_dataset = [];
var collocation_dataset = [];
var colors = ['rgb(255, 99, 132)', 'rgb(255, 159, 64)', 'rgb(138, 174, 20)', 'rgb(53, 197, 226)', 'rgb(196, 49, 226)'];
console.log(word_data);
console.log(collocation_data);
for (var i = 0; i < 5; i++) {
    var word_points = [];
    var collocation_points = [];
    var word_label = word_data[i * xAxisData.length].text;
    var collocation_label = collocation_data[i * xAxisData.length].text;
    for (var x = 0; x < xAxisData.length; x++) {
        word_points.push({x: word_data[i * xAxisData.length + x].year, y: word_data[x].count});
    }

    for (var y = 0; y < xAxisData.length; y++) {
        console.log(i * xAxisData.length + y);
        collocation_points.push({x: collocation_data[i * xAxisData.length + y].year, y: collocation_data[y].count});
    }

    word_dataset.push(
        {
            label: word_label,
            backgroundColor: colors[i],
            borderColor: colors[i],
            data: word_points,
            fill: false,
            lineTension: 0
        }
    );

    collocation_dataset.push(
        {
            label: collocation_label,
            backgroundColor: colors[i],
            borderColor: colors[i],
            data: collocation_points,
            fill: false,
            lineTension: 0
        }
    );

}

new Chart(document.getElementById('word_trend').getContext('2d'),
    {
        type: 'line',
        data: {
            labels: xAxisData,
            datasets: word_dataset,
        },
        options: {
            responsive: true,
            tooltips: {
                mode: 'index',
                intersect: false
            },
            hover: {
                mode: 'nearest',
                intersect: true
            },
            scales: {
                xAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Time'
                    }
                }],
                yAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Frequency'
                    },
                }]
            }
        }
    });

new Chart(document.getElementById('collocation_trend').getContext('2d'),
    {
        type: 'line',
        data: {
            labels: xAxisData,
            datasets: collocation_dataset,
        },
        options: {
            responsive: true,
            tooltips: {
                mode: 'index',
                intersect: false
            },
            hover: {
                mode: 'nearest',
                intersect: true
            },
            scales: {
                xAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Time'
                    }
                }],
                yAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Frequency'
                    },
                }]
            }
        }
    });