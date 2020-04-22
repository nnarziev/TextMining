var filterYearStartList = document.getElementById("filter-year-start");
var filterYearStartSel = filterYearStartList.options[filterYearStartList.selectedIndex].value;
var filterYearEndList = document.getElementById("filter-year-end");
var filterYearEndSel = filterYearEndList.options[filterYearEndList.selectedIndex].value;
var xAxisData = [];

for (var j = filterYearStartSel; j <= filterYearEndSel; j++) {
    xAxisData.push(j);
}

var dataset = [];
var colors = ['rgb(255, 99, 132)', 'rgb(255, 159, 64)', 'rgb(138, 174, 20)', 'rgb(53, 197, 226)', 'rgb(196, 49, 226)'];
console.log(data);
for (var i = 0; i < 5; i++) {
    var points = [];
    var label = data[i * xAxisData.length].text;
    for (var x = 0; x < xAxisData.length; x++) {
        points.push({x: data[i * xAxisData.length + x].year, y: data[x].count});
    }

    dataset.push(
        {
            label: label,
            backgroundColor: colors[i],
            borderColor: colors[i],
            data: points,
            fill: false,
            lineTension: 0
        }
    );

}

new Chart(document.getElementById('trend').getContext('2d'),
    {
        type: 'line',
        data: {
            labels: xAxisData,
            datasets: dataset,
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