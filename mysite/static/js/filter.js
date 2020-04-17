var filterList = document.getElementById("filter");

var filterSel = filterList.options[filterList.selectedIndex].value;

if (filterSel === "year")
    document.getElementById("filter-month").style.display = 'none';
else
    document.getElementById("filter-month").style.display = 'inline-block';

function changeFilterSelection() {
    var filterList = document.getElementById("filter");

    var filterSel = filterList.options[filterList.selectedIndex].value;

    if (filterSel === "year")
        document.getElementById("filter-month").style.display = 'none';
    else
        document.getElementById("filter-month").style.display = 'inline-block';
}